#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include <algorithm>
#include <cmath>

static constexpr int OB = 256;

// ----------------------------------------------------------------------------
// Adam: p -= lr * m_hat / (sqrt(v_hat) + eps)
// AdamW: p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
// (weight decay is applied to parameters directly, NOT to the gradient)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(OB,8)
void adam_step_f32(float*__restrict__ p,const float*__restrict__ g,
    float*__restrict__ m,float*__restrict__ v,
    float lr,float b1,float b2,float eps,float bc1,float bc2,int n){
    int i=blockIdx.x*OB+threadIdx.x;if(i>=n)return;
    float gi=g[i];
    float mi=b1*m[i]+(1.f-b1)*gi;
    float vi=b2*v[i]+(1.f-b2)*gi*gi;
    m[i]=mi;v[i]=vi;
    p[i]-=lr*(mi*bc1)/(sqrtf(vi*bc2)+eps);
}

__global__ __launch_bounds__(OB,8)
void adamw_clipstep_f32(float*__restrict__ p, float*__restrict__ g,
    float*__restrict__ m, float*__restrict__ v,
    const float*__restrict__ gnorm,
    const float*__restrict__ grad_scale_inv,  // 1/scale if scaling, else 1.f
    float clip_max, float weight_decay,
    float lr, float b1, float b2, float eps, float bc1, float bc2,
    int n)
{
    int i=blockIdx.x*OB+threadIdx.x; if(i>=n) return;
    // Unscale gradient first (for FP16 training; scale_inv == 1 disables this).
    float scale_inv = *grad_scale_inv;
    float gi = g[i] * scale_inv;
    // Clip if requested.
    float gn = *gnorm * scale_inv;
    float sc = (clip_max>0.f && gn>clip_max) ? (clip_max/(gn+1e-6f)) : 1.f;
    gi *= sc;
    g[i] = gi;
    // Adam moments.
    float mi = b1*m[i] + (1.f-b1)*gi;
    float vi = b2*v[i] + (1.f-b2)*gi*gi;
    m[i] = mi; v[i] = vi;
    // AdamW update: decoupled weight decay applied to the param itself.
    float pi = p[i];
    float update = lr*(mi*bc1)/(sqrtf(vi*bc2)+eps);
    if(weight_decay > 0.f){
        update += lr * weight_decay * pi;
    }
    p[i] = pi - update;
}

__global__ __launch_bounds__(OB,4)
void grad_norm_sq_f32(const float*__restrict__ g,float*__restrict__ out,int n){
    extern __shared__ float sm[];
    float v=0.f;
    for(int i=blockIdx.x*OB+threadIdx.x;i<n;i+=gridDim.x*OB)v+=g[i]*g[i];
    sm[threadIdx.x]=v;__syncthreads();
    for(int s=OB>>1;s>0;s>>=1){if(threadIdx.x<s)sm[threadIdx.x]+=sm[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)atomicAdd(out,sm[0]);
}
__global__ __launch_bounds__(1)
void sqrt_inplace_f32(float* x){ *x=sqrtf(*x); }

// Detect grad inf/nan (used by GradScaler). Sets out=1 if any element is bad.
__global__ __launch_bounds__(OB)
void grad_inf_detect_f32(const float*__restrict__ g, int*__restrict__ out, int n){
    int i=blockIdx.x*OB+threadIdx.x;
    if(i>=n) return;
    float gi=g[i];
    if(!isfinite(gi)) atomicOr(out,1);
}

__global__ __launch_bounds__(1)
void scale_inv_kernel(const float*__restrict__ scale, float*__restrict__ inv_out){
    *inv_out = 1.f / (*scale);
}

// ----------------------------------------------------------------------------
// AdamGPU: supports both vanilla Adam and AdamW via cfg.
// ----------------------------------------------------------------------------
class AdamGPU {
public:
    AdamGPU(int n, float lr=0.05f, float b1=0.9f, float b2=0.999f, float eps=1e-8f,
            cudaStream_t s=nullptr, AdamWConfig adamw={})
        : n_(n), lr_(lr), b1_(b1), b2_(b2), eps_(eps), stream_(s),
          adamw_(adamw),
          d_m_(n), d_v_(n), d_ns_(1), d_scale_inv_(1){
        d_m_.zero(s); d_v_.zero(s);
        float one=1.f;
        cudaMemcpyAsync(d_scale_inv_.get(), &one, sizeof(float),
            cudaMemcpyHostToDevice, s);
    }
    void step(float* dp, const float* dg){
        ++t_;
        float bc1=1.f/(1.f-std::pow(b1_,t_));
        float bc2=1.f/(1.f-std::pow(b2_,t_));
        int bl=(n_+OB-1)/OB;
        adam_step_f32<<<bl,OB,0,stream_>>>(dp,dg,d_m_.get(),d_v_.get(),
            lr_,b1_,b2_,eps_,bc1,bc2,n_);
    }
    // Graph-friendly, fused: clip + (optional) AdamW + (optional) grad-scale-inv.
    void clip_and_step(float* dp, float* dg, float clip_max,
                       const float* scale_dev=nullptr){
        ++t_;
        float bc1=1.f/(1.f-std::pow(b1_,t_));
        float bc2=1.f/(1.f-std::pow(b2_,t_));
        CUDA_CHECK(cudaMemsetAsync(d_ns_.get(),0,sizeof(float),stream_));
        int bl_red=std::min(2048,(n_+OB-1)/OB);
        grad_norm_sq_f32<<<bl_red,OB,OB*sizeof(float),stream_>>>(dg,d_ns_.get(),n_);
        sqrt_inplace_f32<<<1,1,0,stream_>>>(d_ns_.get());
        // Compute 1/scale on device if a scaler is in use.
        if(scale_dev){
            scale_inv_kernel<<<1,1,0,stream_>>>(scale_dev, d_scale_inv_.get());
        }
        int bl=(n_+OB-1)/OB;
        float wd = adamw_.enabled ? adamw_.weight_decay : 0.f;
        adamw_clipstep_f32<<<bl,OB,0,stream_>>>(dp, dg, d_m_.get(), d_v_.get(),
            d_ns_.get(), d_scale_inv_.get(),
            clip_max, wd, lr_, b1_, b2_, eps_, bc1, bc2, n_);
    }
    void set_lr(float lr){lr_=lr;}
    float lr()const{return lr_;}
    void reset(){t_=0; d_m_.zero(stream_); d_v_.zero(stream_);}
private:
    int n_, t_=0;
    float lr_,b1_,b2_,eps_;
    cudaStream_t stream_;
    AdamWConfig adamw_;
    DeviceBuffer<float> d_m_,d_v_,d_ns_;
    DeviceBuffer<float> d_scale_inv_;
};

// ----------------------------------------------------------------------------
// GradScaler: dynamic loss scaling. Multiplies loss by `scale` before backward
// so FP16 gradients don't underflow; AdamGPU::clip_and_step unscales before
// applying. If a step had any inf/nan, scale halves; otherwise scale doubles
// every growth_interval clean steps.
// ----------------------------------------------------------------------------
class GradScaler {
public:
    GradScaler(int n, GradScalerConfig cfg, cudaStream_t s=nullptr)
        : cfg_(cfg), n_(n), stream_(s), d_scale_(1), d_inf_flag_(1){
        if(!cfg_.enabled) return;
        cudaMemcpyAsync(d_scale_.get(), &cfg_.init_scale, sizeof(float),
            cudaMemcpyHostToDevice, s);
        cudaMemsetAsync(d_inf_flag_.get(), 0, sizeof(int), s);
    }
    bool enabled() const { return cfg_.enabled; }
    float* scale_ptr(){ return d_scale_.get(); }

    // After backward, before optimizer.step(): check for inf/nan in d_grads.
    // If found, skip step and reduce scale.
    void detect_and_update(const float* d_grads){
        if(!cfg_.enabled) return;
        cudaMemsetAsync(d_inf_flag_.get(), 0, sizeof(int), stream_);
        int bl=(n_+OB-1)/OB;
        grad_inf_detect_f32<<<bl,OB,0,stream_>>>(d_grads, d_inf_flag_.get(), n_);
        int flag;
        cudaMemcpyAsync(&flag, d_inf_flag_.get(), sizeof(int),
            cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        float scale;
        cudaMemcpyAsync(&scale, d_scale_.get(), sizeof(float),
            cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        if(flag){
            scale *= cfg_.backoff_factor;
            clean_streak_ = 0;
            should_step_  = false;
        } else {
            ++clean_streak_;
            if(clean_streak_ >= cfg_.growth_interval){
                scale *= cfg_.growth_factor;
                clean_streak_ = 0;
            }
            should_step_ = true;
        }
        cudaMemcpyAsync(d_scale_.get(), &scale, sizeof(float),
            cudaMemcpyHostToDevice, stream_);
    }
    bool should_step() const { return cfg_.enabled ? should_step_ : true; }

private:
    GradScalerConfig cfg_;
    int n_;
    cudaStream_t stream_;
    DeviceBuffer<float> d_scale_;
    DeviceBuffer<int>   d_inf_flag_;
    int  clean_streak_=0;
    bool should_step_=true;
};

class LRScheduler {
public:
    enum class Mode{CONSTANT,COSINE,STEP,WARMUP_COSINE};
    LRScheduler(float lr,int total,Mode m=Mode::COSINE,float lrmin=0.f,int warm=0)
        :lr_(lr),lrmin_(lrmin),total_(total),warm_(warm),mode_(m){}
    float get(int step)const{
        if(mode_==Mode::CONSTANT)return lr_;
        if(step<warm_)return lr_*(float)step/std::max(warm_,1);
        float p=std::min((float)(step-warm_)/std::max(total_-warm_,1),1.f);
        switch(mode_){
            case Mode::COSINE:case Mode::WARMUP_COSINE:
                return lrmin_+.5f*(lr_-lrmin_)*(1.f+cosf(3.14159265f*p));
            case Mode::STEP:return lr_*powf(.5f,(float)(step/(total_/4)));
            default:return lr_;
        }
    }
private:
    float lr_,lrmin_;int total_,warm_;Mode mode_;
};
