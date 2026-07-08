#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include "QHPC_reduction_v19.hpp"
#include "QHPC_async_pipeline_v19.hpp"
#include "QHPC_vectorized_kernels_v19.cuh"
#include "QHPC_tpb_v19.hpp"
#include "QHPC_chain_1q_v19.cuh"
#include <algorithm>
#include <cassert>

static constexpr int BSZ = 256;

// Param-baking is still solved by reading θ from d_params[param_idx] inside
// each kernel. v15 adds: gradient writes go to ReductionScratch (no atomics),
// then a reducer kernel folds blocks -> d_grads at the end of backward.

// v19b: Graph-safe basis-state initialiser. Replaces the v15-v19 use of
// cudaMemcpyAsync from a host stack `float2 one={1,0}` that was invisible
// to the graph capture machinery and produced a silent zero-filled state on
// graph replay (manifesting as best=0 in the v19 bench).
__global__ __launch_bounds__(1)
void _write_basis_state_kernel(float2* psi){
    psi[0] = make_float2(1.0f, 0.0f);
}

__global__ __launch_bounds__(BSZ,8)
void sv_apply_1q_static_f32(float2*__restrict__ sv,
    float g00r,float g00i,float g01r,float g01i,
    float g10r,float g10i,float g11r,float g11i,
    int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 a=sv[i0],b=sv[i1];
    sv[i0]={g00r*a.x-g00i*a.y+g01r*b.x-g01i*b.y,g00r*a.y+g00i*a.x+g01r*b.y+g01i*b.x};
    sv[i1]={g10r*a.x-g10i*a.y+g11r*b.x-g11i*b.y,g10r*a.y+g10i*a.x+g11r*b.y+g11i*b.x};
}

__global__ __launch_bounds__(BSZ,8)
void sv_apply_rx_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x+s*p1.y, c*p0.y-s*p1.x};
    sv[i1]={s*p0.y+c*p1.x,-s*p0.x+c*p1.y};
}
__global__ __launch_bounds__(BSZ,8)
void sv_apply_ry_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x-s*p1.x, c*p0.y-s*p1.y};
    sv[i1]={s*p0.x+c*p1.x, s*p0.y+c*p1.y};
}
__global__ __launch_bounds__(BSZ,8)
void sv_apply_rz_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x+s*p0.y, c*p0.y-s*p0.x};
    sv[i1]={c*p1.x-s*p1.y, c*p1.y+s*p1.x};
}

__global__ __launch_bounds__(BSZ,8)
void sv_apply_rx_inv_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c); s=-s;
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x+s*p1.y, c*p0.y-s*p1.x};
    sv[i1]={s*p0.y+c*p1.x,-s*p0.x+c*p1.y};
}
__global__ __launch_bounds__(BSZ,8)
void sv_apply_ry_inv_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c); s=-s;
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x-s*p1.x, c*p0.y-s*p1.y};
    sv[i1]={s*p0.x+c*p1.x, s*p0.y+c*p1.y};
}
__global__ __launch_bounds__(BSZ,8)
void sv_apply_rz_inv_dyn_f32(float2*__restrict__ sv,
    const float*__restrict__ d_params,int param_idx,int q,long long np){
    long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(t>=np)return;
    float theta=d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c); s=-s;
    long long lo=t&((1LL<<q)-1),hi=t>>q;
    long long i0=(hi<<(q+1))|lo,i1=i0|(1LL<<q);
    float2 p0=sv[i0],p1=sv[i1];
    sv[i0]={c*p0.x+s*p0.y, c*p0.y-s*p0.x};
    sv[i1]={c*p1.x-s*p1.y, c*p1.y+s*p1.x};
}

__global__ __launch_bounds__(BSZ,8)
void sv_cnot_f32(float2*__restrict__ sv,int ctrl,int tgt,long long N){
    long long i=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(i>=N)return;
    if(!((i>>ctrl)&1))return;
    long long j=i^(1LL<<tgt);
    if(j>i){float2 t=sv[i];sv[i]=sv[j];sv[j]=t;}
}
__global__ __launch_bounds__(BSZ,8)
void sv_cz_f32(float2*__restrict__ sv,int ctrl,int tgt,long long N){
    long long i=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(i>=N)return;
    if(((i>>ctrl)&1)&&((i>>tgt)&1))sv[i]={-sv[i].x,-sv[i].y};
}

// ============================================================================
// FUSED grad + un-compute kernels — v15 atomic-free.
// Each block writes its per-block partial sum into partials[blockIdx.x, param_idx].
// No atomicAdd. A separate reducer kernel folds blocks -> d_grads after the
// whole backward pass.
// ============================================================================

__global__ __launch_bounds__(BSZ,4)
void grad_uncomp_rz_dyn_v15(float2*__restrict__ psi, float2*__restrict__ phi,
    float* __restrict__ partials, int n_params,
    const float*__restrict__ d_params, int param_idx,
    int q, long long np)
{
    float theta = d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    float gl = 0.f;
    for(long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
        t<np; t+=(long long)gridDim.x*BSZ){
        long long lo=t&((1LL<<q)-1), hi=t>>q;
        long long i0=(hi<<(q+1))|lo, i1=i0|(1LL<<q);
        float2 p0=psi[i0],p1=psi[i1],f0=phi[i0],f1=phi[i1];
        gl += (f0.x*p0.y - f0.y*p0.x) - (f1.x*p1.y - f1.y*p1.x);
        psi[i0]={c*p0.x-s*p0.y, c*p0.y+s*p0.x};
        psi[i1]={c*p1.x+s*p1.y, c*p1.y-s*p1.x};
        phi[i0]={c*f0.x-s*f0.y, c*f0.y+s*f0.x};
        phi[i1]={c*f1.x+s*f1.y, c*f1.y-s*f1.x};
    }
    gl = qhpc_block_sum(gl);
    qhpc_store_block_partial(partials, n_params, param_idx, gl);
}

__global__ __launch_bounds__(BSZ,4)
void grad_uncomp_rx_dyn_v15(float2*__restrict__ psi, float2*__restrict__ phi,
    float* __restrict__ partials, int n_params,
    const float*__restrict__ d_params, int param_idx,
    int q, long long np)
{
    float theta = d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    float gl = 0.f;
    for(long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
        t<np; t+=(long long)gridDim.x*BSZ){
        long long lo=t&((1LL<<q)-1), hi=t>>q;
        long long i0=(hi<<(q+1))|lo, i1=i0|(1LL<<q);
        float2 p0=psi[i0],p1=psi[i1],f0=phi[i0],f1=phi[i1];
        gl += (f0.x*p1.y - f0.y*p1.x) + (f1.x*p0.y - f1.y*p0.x);
        psi[i0]={c*p0.x-s*p1.y, c*p0.y+s*p1.x};
        psi[i1]={-s*p0.y+c*p1.x, s*p0.x+c*p1.y};
        phi[i0]={c*f0.x-s*f1.y, c*f0.y+s*f1.x};
        phi[i1]={-s*f0.y+c*f1.x, s*f0.x+c*f1.y};
    }
    gl = qhpc_block_sum(gl);
    qhpc_store_block_partial(partials, n_params, param_idx, gl);
}

__global__ __launch_bounds__(BSZ,4)
void grad_uncomp_ry_dyn_v15(float2*__restrict__ psi, float2*__restrict__ phi,
    float* __restrict__ partials, int n_params,
    const float*__restrict__ d_params, int param_idx,
    int q, long long np)
{
    float theta = d_params[param_idx];
    float c,s; __sincosf(theta*0.5f,&s,&c);
    float gl = 0.f;
    for(long long t=(long long)blockIdx.x*BSZ+threadIdx.x;
        t<np; t+=(long long)gridDim.x*BSZ){
        long long lo=t&((1LL<<q)-1), hi=t>>q;
        long long i0=(hi<<(q+1))|lo, i1=i0|(1LL<<q);
        float2 p0=psi[i0],p1=psi[i1],f0=phi[i0],f1=phi[i1];
        gl += (f1.x*p0.x + f1.y*p0.y) - (f0.x*p1.x + f0.y*p1.y);
        psi[i0]={c*p0.x+s*p1.x, c*p0.y+s*p1.y};
        psi[i1]={-s*p0.x+c*p1.x,-s*p0.y+c*p1.y};
        phi[i0]={c*f0.x+s*f1.x, c*f0.y+s*f1.y};
        phi[i1]={-s*f0.x+c*f1.x,-s*f0.y+c*f1.y};
    }
    gl = qhpc_block_sum(gl);
    qhpc_store_block_partial(partials, n_params, param_idx, gl);
}

__global__ __launch_bounds__(BSZ,4)
void sv_pauli_axpy_f32(float2*__restrict__ out,const float2*__restrict__ inp,
    uint32_t xm,uint32_t ym,uint32_t zm,float coeff,long long N){
    long long i=(long long)blockIdx.x*BSZ+threadIdx.x;
    if(i>=N)return;
    long long j=i^(long long)(xm|ym);
    int zp=__popc((uint32_t)i&zm)&1;
    int yp=__popc((uint32_t)i&ym)&1;
    float sg=(zp^yp)?-1.f:1.f;
    int ny=__popc(ym);
    float2 src=inp[j],res;
    switch(ny&3){
        case 0:res={ sg*src.x, sg*src.y};break;
        case 1:res={-sg*src.y, sg*src.x};break;
        case 2:res={-sg*src.x,-sg*src.y};break;
        default:res={ sg*src.y,-sg*src.x};break;
    }
    out[i].x+=coeff*res.x;
    out[i].y+=coeff*res.y;
}

__global__ __launch_bounds__(BSZ,4)
void sv_expectation_fused_f32(const float2*__restrict__ sv,
    const GPUPauliTerm*__restrict__ terms,int n_terms,
    float*__restrict__ out,long long N)
{
    float val=0.f;
    for(long long i=(long long)blockIdx.x*BSZ+threadIdx.x;
        i<N; i+=(long long)gridDim.x*BSZ){
        float local=0.f;
        for(int t=0;t<n_terms;++t){
            GPUPauliTerm pt=terms[t];
            long long j=i^(long long)(pt.x_mask|pt.y_mask);
            int zp=__popc((uint32_t)i&pt.z_mask)&1;
            int yp=__popc((uint32_t)i&pt.y_mask)&1;
            float sg=(zp^yp)?-1.f:1.f;
            int ny=__popc(pt.y_mask);
            float2 a=sv[i],b=sv[j];
            float c;
            switch(ny&3){
                case 0:c= sg*(a.x*b.x+a.y*b.y);break;
                case 1:c= sg*(a.x*b.y-a.y*b.x);break;
                case 2:c=-sg*(a.x*b.x+a.y*b.y);break;
                default:c=-sg*(a.x*b.y-a.y*b.x);break;
            }
            local+=pt.coeff*c;
        }
        val+=local;
    }
    val = qhpc_block_sum(val);
    // Expectation is a SCALAR, so a single atomicAdd here is acceptable
    // (only n_blocks contend, not n_blocks * n_params). Replaceable with
    // ReductionScratch[1] if even this is too much.
    if(threadIdx.x==0) atomicAdd(out, val);
}

// ============================================================================
// Engine — v15 with ReductionScratch ownership.
// ============================================================================
class TrueAdjointDiffEngine {
public:
    TrueAdjointDiffEngine(const QMLCircuit& circ,const Observable& obs,
                          cudaStream_t stream=nullptr,bool use_fused=true,
                          bool use_tpb=false)
        :circ_(circ),obs_(obs),stream_(stream),use_fused_(use_fused),
         use_tpb_(use_tpb)
    {
        n_states_=1LL<<circ.n_qubits;
        d_psi_.resize(n_states_);
        d_phi_.resize(n_states_);
        d_loss_.resize(1);
        d_obs_.resize(obs.terms.size());
        if(!obs.terms.empty())
            d_obs_.upload(obs.terms.data(),(int)obs.terms.size(),stream);
        sched_=GateSchedule::build(circ_);

        // Sizing: max grid we'll launch for the fused grad kernels.
        int max_grid = (int)std::min(2048LL, (n_states_/2 + BSZ - 1) / BSZ);
        scratch_.resize(max_grid, std::max(1, circ_.n_params));
        max_grid_ = max_grid;

        // ---- v18 TPB grouping setup ----
        // Partition the observable into commuting cliques. Each clique is a
        // contiguous block of terms in d_obs_clique_terms_. The expectation
        // kernel walks ONE clique per launch, reading ψ[i] once per thread
        // and accumulating contributions from all K terms in the clique.
        if(use_tpb_ && !obs.terms.empty()){
            auto groups = tpb_partition(obs);
            // Flatten:  [terms of clique 0][terms of clique 1]...
            std::vector<GPUPauliTerm> flat;
            flat.reserve(obs.terms.size());
            clique_offsets_.clear();
            clique_offsets_.push_back(0);
            for(const auto& g : groups){
                for(int idx : g.term_indices) flat.push_back(obs.terms[idx]);
                clique_offsets_.push_back((int)flat.size());
            }
            d_obs_clique_terms_.resize(flat.size());
            if(!flat.empty())
                d_obs_clique_terms_.upload(flat.data(), (int)flat.size(), stream);
            n_cliques_ = (int)groups.size();
        }

        // ---- v19: build chained 1q-rotation schedule ----
        // Detect maximal runs of 1q-param gates on the same qubit with no
        // intervening gate touching that qubit. Each run becomes a single
        // launch via the chain kernel — fewer launches, fewer VRAM round
        // trips.
        sched_fwd_chained_ = build_chained_schedule(sched_.fwd, /*inverse=*/false);
        sched_bwd_chained_ = build_chained_schedule(sched_.bwd, /*inverse=*/true);

        // Flatten chain metadata for device upload. We pack ALL chains'
        // (param_idx, axes, signs) into 3 long contiguous arrays and store
        // per-chain offsets.
        std::vector<int>  flat_pidx_fwd;
        std::vector<char> flat_axes_fwd, flat_signs_fwd;
        std::vector<int>  off_fwd;
        for(const auto& s : sched_fwd_chained_){
            off_fwd.push_back((int)flat_pidx_fwd.size());
            if(s.kind == SchedStep::CHAIN_1Q){
                for(int k=0;k<s.chain.len;++k){
                    flat_pidx_fwd.push_back(s.chain.param_idx[k]);
                    flat_axes_fwd.push_back(s.chain.axes[k]);
                    flat_signs_fwd.push_back(s.chain.signs[k]);
                }
            }
        }
        std::vector<int>  flat_pidx_bwd;
        std::vector<char> flat_axes_bwd, flat_signs_bwd;
        std::vector<int>  off_bwd;
        for(const auto& s : sched_bwd_chained_){
            off_bwd.push_back((int)flat_pidx_bwd.size());
            if(s.kind == SchedStep::CHAIN_1Q){
                for(int k=0;k<s.chain.len;++k){
                    flat_pidx_bwd.push_back(s.chain.param_idx[k]);
                    flat_axes_bwd.push_back(s.chain.axes[k]);
                    flat_signs_bwd.push_back(s.chain.signs[k]);
                }
            }
        }

        if(!flat_pidx_fwd.empty()){
            d_chain_pidx_fwd_.resize(flat_pidx_fwd.size());
            d_chain_axes_fwd_.resize(flat_axes_fwd.size());
            d_chain_signs_fwd_.resize(flat_signs_fwd.size());
            d_chain_pidx_fwd_.upload(flat_pidx_fwd.data(), (int)flat_pidx_fwd.size(), stream);
            d_chain_axes_fwd_.upload(flat_axes_fwd.data(), (int)flat_axes_fwd.size(), stream);
            d_chain_signs_fwd_.upload(flat_signs_fwd.data(), (int)flat_signs_fwd.size(), stream);
        }
        if(!flat_pidx_bwd.empty()){
            d_chain_pidx_bwd_.resize(flat_pidx_bwd.size());
            d_chain_axes_bwd_.resize(flat_axes_bwd.size());
            d_chain_signs_bwd_.resize(flat_signs_bwd.size());
            d_chain_pidx_bwd_.upload(flat_pidx_bwd.data(), (int)flat_pidx_bwd.size(), stream);
            d_chain_axes_bwd_.upload(flat_axes_bwd.data(), (int)flat_axes_bwd.size(), stream);
            d_chain_signs_bwd_.upload(flat_signs_bwd.data(), (int)flat_signs_bwd.size(), stream);
        }
        chain_off_fwd_ = std::move(off_fwd);
        chain_off_bwd_ = std::move(off_bwd);

        int n_chains_fwd = 0, n_chain_gates_fwd = 0;
        for(const auto& s : sched_fwd_chained_){
            if(s.kind == SchedStep::CHAIN_1Q){
                ++n_chains_fwd; n_chain_gates_fwd += s.chain.len;
            }
        }
        if(n_chains_fwd > 0){
            fprintf(stderr,
                "[Chain1q] fwd: %d 1q-param gates packed into %d chains "
                "(avg chain len = %.2f) -> %dx fewer launches\n",
                n_chain_gates_fwd, n_chains_fwd,
                (double)n_chain_gates_fwd / n_chains_fwd,
                (n_chains_fwd > 0 ? n_chain_gates_fwd / n_chains_fwd : 1));
        }
    }

    void forward_dev(const float* d_params){
        _reset();_fwd(d_params);
        _zero_loss();
        _exp_fused();
    }

    void forward_backward_dev(const float* d_params, float* dg){
        CUDA_CHECK(cudaMemsetAsync(dg, 0, circ_.n_params*sizeof(float), stream_));
        // Zero the per-block partials buffer ONCE per backward.
        scratch_.zero(stream_);
        _reset(); _fwd(d_params);
        _zero_loss(); _exp_fused();
        _init_phi();
        _bwd(d_params);
        // Fold partials -> d_grads (single launch, no atomics).
        reduce_partials_to_grad(scratch_, dg, max_grid_, stream_);
    }

    // Materialise per-parameter derivative vectors  |∂_i ψ(θ)⟩  into the
    // QNG engine's mu buffer.  Uses the parameter-shift rule:
    //
    //     |∂_i ψ⟩ ≈ ( ψ(θ_i + π/2) − ψ(θ_i − π/2) ) / 2
    //
    // Costs 2·P forward passes — non-trivial, but QNG typically replaces
    // hundreds of vanilla-grad epochs, so the trade is a massive net win.
    void materialise_param_derivatives(const float* d_params,
                                       float2* d_mu_pn,
                                       long long mu_stride_elems);

    float forward(const float* hp){
        DeviceBuffer<float> dp(circ_.n_params);
        CUDA_CHECK(cudaMemcpyAsync(dp.get(), hp, circ_.n_params*sizeof(float),
            cudaMemcpyHostToDevice, stream_));
        forward_dev(dp.get());
        float E;
        CUDA_CHECK(cudaMemcpyAsync(&E, d_loss_.get(), sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return E;
    }

    float* get_loss_ptr(){ return d_loss_.get(); }
    float2* get_psi_ptr(){ return d_psi_.get(); }
    long long n_states()const{ return n_states_; }

private:
    const QMLCircuit& circ_;
    const Observable& obs_;
    cudaStream_t stream_;
    bool use_fused_;
    bool use_tpb_ = false;
    long long n_states_;
    GateSchedule sched_;
    // v19 chained 1q schedules + flattened device metadata.
    std::vector<SchedStep>     sched_fwd_chained_;
    std::vector<SchedStep>     sched_bwd_chained_;
    std::vector<int>           chain_off_fwd_;
    std::vector<int>           chain_off_bwd_;
    DeviceBuffer<int>          d_chain_pidx_fwd_;
    DeviceBuffer<char>         d_chain_axes_fwd_;
    DeviceBuffer<char>         d_chain_signs_fwd_;
    DeviceBuffer<int>          d_chain_pidx_bwd_;
    DeviceBuffer<char>         d_chain_axes_bwd_;
    DeviceBuffer<char>         d_chain_signs_bwd_;
    DeviceBuffer<float2>       d_psi_, d_phi_;
    DeviceBuffer<float>        d_loss_;
    DeviceBuffer<GPUPauliTerm> d_obs_;
    DeviceBuffer<GPUPauliTerm> d_obs_clique_terms_;   // flattened TPB layout
    std::vector<int>           clique_offsets_;       // [0, K0, K0+K1, ...]
    int                        n_cliques_ = 0;
    ReductionScratch           scratch_;
    int max_grid_=0;

    int BLK(long long n)const{return (int)std::min(2048LL, (n+BSZ-1)/BSZ);}

    void _reset(){
        CUDA_CHECK(cudaMemsetAsync(d_psi_.get(), 0, n_states_*sizeof(float2), stream_));
        // Set amplitude[0] = (1,0). MUST NOT use cudaMemcpyAsync from a host
        // stack variable here: when this function is recorded into a CUDA
        // Graph, the captured Memcpy will reference a stack address that is
        // invalid by replay time, producing zero-filled state and the
        // false-speedup we observed in v19. Launch a 1-thread kernel that
        // writes the basis-state constant directly on the device.
        _write_basis_state_kernel<<<1,1,0,stream_>>>(d_psi_.get());
    }
    void _zero_loss(){
        CUDA_CHECK(cudaMemsetAsync(d_loss_.get(), 0, sizeof(float), stream_));
    }

    void _apply_fwd_dyn(float2* sv, const ScheduledOp& op, const float* d_params){
        long long np=n_states_>>1; int blp=BLK(np); int blN=BLK(n_states_);
        switch(op.kind){
            case ScheduledOp::GATE_1Q_STATIC:{
                Gate1qMatrix M;
                switch(op.type){
                    case GateType::H:M=gate_h();break;
                    case GateType::S:M=gate_s();break;
                    case GateType::T:M=gate_t();break;
                    default:M=gate_h();break;
                }
                sv_apply_1q_static_f32<<<blp,BSZ,0,stream_>>>(sv,
                    M.g[0],M.g[1],M.g[2],M.g[3],M.g[4],M.g[5],M.g[6],M.g[7],
                    op.q0,np); break;}
            case ScheduledOp::GATE_1Q_PARAM:
                // v16 vectorized: q==0 uses contiguous float4 LD/ST.E.128;
                // q>=1 uses the two-pairs-per-thread variant — both achieve
                // 128-bit memory transactions, nearly doubling throughput on
                // bandwidth-bound 1q-param kernels (the common hot path).
                if(op.type==GateType::RX)
                    qhpc_v16::launch_sv_apply_rx_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/false);
                else if(op.type==GateType::RY)
                    qhpc_v16::launch_sv_apply_ry_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/false);
                else
                    qhpc_v16::launch_sv_apply_rz_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/false);
                break;
            case ScheduledOp::GATE_CNOT:
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_); break;
            case ScheduledOp::GATE_CZ:
                sv_cz_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_); break;
            case ScheduledOp::GATE_CRZ:
                // CNOT - RZ_dyn - CNOT decomposition (simple).
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_);
                qhpc_v16::launch_sv_apply_rz_v16(sv, d_params, op.param_idx, op.q1, np, stream_, false);
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_);
                break;
        }
    }

    void _apply_adj_dyn(float2* sv, const ScheduledOp& op, const float* d_params){
        long long np=n_states_>>1; int blp=BLK(np); int blN=BLK(n_states_);
        switch(op.kind){
            case ScheduledOp::GATE_1Q_STATIC:{
                Gate1qMatrix M;
                switch(op.type){
                    case GateType::H: M=gate_h();break;
                    case GateType::S: M={1,0,0,0,0,0,0,-1};break;
                    case GateType::T:{static constexpr float r=0.70710678118f;
                                     M={1,0,0,0,0,0,r,-r};break;}
                    default:M=gate_h();break;
                }
                sv_apply_1q_static_f32<<<blp,BSZ,0,stream_>>>(sv,
                    M.g[0],M.g[1],M.g[2],M.g[3],M.g[4],M.g[5],M.g[6],M.g[7],
                    op.q0,np); break;}
            case ScheduledOp::GATE_1Q_PARAM:
                if(op.type==GateType::RX)
                    qhpc_v16::launch_sv_apply_rx_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/true);
                else if(op.type==GateType::RY)
                    qhpc_v16::launch_sv_apply_ry_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/true);
                else
                    qhpc_v16::launch_sv_apply_rz_v16(sv, d_params, op.param_idx, op.q0, np, stream_, /*inverse=*/true);
                break;
            case ScheduledOp::GATE_CNOT:
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_); break;
            case ScheduledOp::GATE_CZ:
                sv_cz_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_); break;
            case ScheduledOp::GATE_CRZ:
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_);
                qhpc_v16::launch_sv_apply_rz_v16(sv, d_params, op.param_idx, op.q1, np, stream_, true);
                sv_cnot_f32<<<blN,BSZ,0,stream_>>>(sv,op.q0,op.q1,n_states_);
                break;
        }
    }

    void _fwd(const float* d_params){
        if(sched_fwd_chained_.empty()){
            // Fallback to per-gate dispatch.
            for(const auto& op : sched_.fwd) _apply_fwd_dyn(d_psi_.get(), op, d_params);
            return;
        }
        const long long np = n_states_ >> 1;
        for(size_t i = 0; i < sched_fwd_chained_.size(); ++i){
            const auto& s = sched_fwd_chained_[i];
            if(s.kind == SchedStep::CHAIN_1Q){
                int off = chain_off_fwd_[i];
                qhpc_v19::launch_chain_1q_v19(
                    d_psi_.get(), d_params,
                    d_chain_pidx_fwd_.get() + off,
                    d_chain_axes_fwd_.get() + off,
                    d_chain_signs_fwd_.get() + off,
                    s.chain.len, s.chain.q, np, stream_);
            } else {
                _apply_fwd_dyn(d_psi_.get(), s.op, d_params);
            }
        }
    }
    void _exp_fused(){
        if(obs_.terms.empty()) return;
        long long bl=BLK(n_states_);
        if(use_tpb_ && n_cliques_ > 0){
            // One kernel launch per clique. Each launch reads ψ ONCE and
            // accumulates expectations of all K terms in the clique.
            // For O(10⁴)-term chemistry Hamiltonians this collapses to
            // O(10²) sweeps — a ~100× reduction in VRAM traffic.
            for(int c = 0; c < n_cliques_; ++c){
                int start = clique_offsets_[c];
                int end   = clique_offsets_[c+1];
                int K = end - start;
                if(K == 0) continue;
                sv_expectation_tpb_clique_f32<<<bl, BSZ, 0, stream_>>>(
                    d_psi_.get(),
                    d_obs_clique_terms_.get() + start,
                    K,
                    d_loss_.get(),
                    n_states_);
            }
        } else {
            sv_expectation_fused_f32<<<bl,BSZ,0,stream_>>>(
                d_psi_.get(), d_obs_.get(), (int)obs_.terms.size(),
                d_loss_.get(), n_states_);
        }
    }
    void _init_phi(){
        CUDA_CHECK(cudaMemsetAsync(d_phi_.get(), 0, n_states_*sizeof(float2), stream_));
        long long bl=BLK(n_states_);
        for(const auto& pt : obs_.terms)
            sv_pauli_axpy_f32<<<bl,BSZ,0,stream_>>>(
                d_phi_.get(), d_psi_.get(),
                pt.x_mask, pt.y_mask, pt.z_mask, pt.coeff, n_states_);
    }

    void _bwd(const float* d_params){
        const long long np = n_states_ >> 1;
        const int blp = BLK(np);
        for(const auto& op : sched_.bwd){
            if(op.kind == ScheduledOp::GATE_1Q_PARAM && use_fused_){
                switch(op.type){
                    case GateType::RX:
                        grad_uncomp_rx_dyn_v15<<<max_grid_,BSZ,0,stream_>>>(
                            d_psi_.get(), d_phi_.get(),
                            scratch_.partials.get(), scratch_.n_params,
                            d_params, op.param_idx, op.q0, np);
                        break;
                    case GateType::RY:
                        grad_uncomp_ry_dyn_v15<<<max_grid_,BSZ,0,stream_>>>(
                            d_psi_.get(), d_phi_.get(),
                            scratch_.partials.get(), scratch_.n_params,
                            d_params, op.param_idx, op.q0, np);
                        break;
                    case GateType::RZ:
                        grad_uncomp_rz_dyn_v15<<<max_grid_,BSZ,0,stream_>>>(
                            d_psi_.get(), d_phi_.get(),
                            scratch_.partials.get(), scratch_.n_params,
                            d_params, op.param_idx, op.q0, np);
                        break;
                    default: break;
                }
                (void)blp;
            } else {
                _apply_adj_dyn(d_psi_.get(), op, d_params);
                _apply_adj_dyn(d_phi_.get(), op, d_params);
            }
        }
    }
};

// (mu_p ← (mu_p − psi) * 0.5) — kernel used by parameter-shift derivative.
__global__ __launch_bounds__(256)
void qhpc_saxpy_mu_minus_kernel(float2* mu, const float2* psi, long long N){
    long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if(i >= N) return;
    float2 m = mu[i], p = psi[i];
    mu[i] = make_float2(0.5f * (m.x - p.x), 0.5f * (m.y - p.y));
}

// Device kernel: write d_out[p] = d_src[p] + delta, leaving all other elements unchanged.
// Used by the QNG materialise loop to shift exactly one parameter at a time.
__global__ __launch_bounds__(1)
void qhpc_shift_param_kernel(const float* src, float* dst, int p, float delta){
    dst[p] = src[p] + delta;
}

inline void TrueAdjointDiffEngine::materialise_param_derivatives(
    const float* d_params, float2* d_mu_pn, long long mu_stride_elems)
{
    constexpr float PI_2 = 1.5707963267948966f;
    DeviceBuffer<float> dp_plus(circ_.n_params);
    DeviceBuffer<float> dp_minus(circ_.n_params);
    // Initial copy: dp_plus = dp_minus = d_params.
    CUDA_CHECK(cudaMemcpyAsync(dp_plus.get(),  d_params,
        circ_.n_params * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dp_minus.get(), d_params,
        circ_.n_params * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

    int prev_p = -1;
    for(int p = 0; p < circ_.n_params; ++p){
        // Restore previous slot before patching the new one — keeps dp_plus
        // / dp_minus equal to d_params everywhere except element p.
        if(prev_p >= 0){
            CUDA_CHECK(cudaMemcpyAsync(dp_plus.get()  + prev_p, d_params + prev_p,
                sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(dp_minus.get() + prev_p, d_params + prev_p,
                sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        qhpc_shift_param_kernel<<<1,1,0,stream_>>>(d_params, dp_plus.get(),  p,  PI_2);
        qhpc_shift_param_kernel<<<1,1,0,stream_>>>(d_params, dp_minus.get(), p, -PI_2);

        _reset(); _fwd(dp_plus.get());
        CUDA_CHECK(cudaMemcpyAsync(
            d_mu_pn + (size_t)p * mu_stride_elems,
            d_psi_.get(),
            (size_t)n_states_ * sizeof(float2),
            cudaMemcpyDeviceToDevice, stream_));

        _reset(); _fwd(dp_minus.get());
        int blocks = (int)((n_states_ + 255) / 256);
        qhpc_saxpy_mu_minus_kernel<<<blocks, 256, 0, stream_>>>(
            d_mu_pn + (size_t)p * mu_stride_elems,
            d_psi_.get(),
            n_states_);
        prev_p = p;
    }
}


