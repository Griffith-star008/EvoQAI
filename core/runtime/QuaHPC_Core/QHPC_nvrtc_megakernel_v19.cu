#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include <nvrtc.h>
#include <cuda.h>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>

#define NVRTC_CHECK(x) do{ nvrtcResult _r=(x); if(_r!=NVRTC_SUCCESS){ \
    fprintf(stderr,"[NVRTC] %s\n",nvrtcGetErrorString(_r)); std::abort(); } }while(0)
#define CUR_CHECK(x) do{ CUresult _r=(x); if(_r!=CUDA_SUCCESS){ \
    const char* es=nullptr; cuGetErrorString(_r,&es); \
    fprintf(stderr,"[CU] %s\n",es?es:"?"); std::abort(); } }while(0)

// Mega-kernel: state vector resides entirely in shared memory while ALL gates run
// in registers/shared. Forward + expectation in one launch. Supports up to ~12
// qubits (4096 amplitudes * 8 bytes = 32 KB shared mem). For larger n a tiled
// variant is needed (sketched in build_megakernel_large).
class NVRTCMegaKernelEngine {
public:
    // HARD CEILING:
    //   N=13 -> 64 KB shared mem (fits on Ampere/Ada with opt-in dyn smem)
    //   N=14 -> 128 KB > 99 KB Ada hardware max -> REFUSE and signal fallback.
    // Caller must check ok() and dispatch to STATEVEC_KERNELS if false.
    NVRTCMegaKernelEngine(const QMLCircuit& circ, const Observable& obs,
                          cudaStream_t stream=nullptr, const NVRTCConfig& cfg={})
        :circ_(circ), obs_(obs), stream_(stream), cfg_(cfg){
        n_ = circ.n_qubits;
        n_states_ = 1LL << n_;
        n_terms_ = (int)obs_.terms.size();

        size_t need = (size_t)n_states_ * sizeof(float2);
        if(need > 96 * 1024){
            fprintf(stderr,
                "[NVRTC] n_qubits=%d needs %zu B shared mem (>96KB). "
                "Refusing to build; use STATEVEC_KERNELS instead.\n",
                n_, need);
            build_ok_ = false;
            return;
        }
        if(need > 48 * 1024){
            need_optin_smem_ = true;
            dyn_smem_bytes_  = (int)need;
        }

        d_loss_.resize(1);
        d_psi_.resize(n_states_);
        _upload_obs();
        _detect_cc();
        _build_and_load();   // COMPILE-ONCE: never invoke from training hot loop.
        build_ok_ = (func_fwd_ != nullptr);
    }

    bool ok() const { return build_ok_; }

    void forward_dev(const float* d_params){
        if(!build_ok_){
            fprintf(stderr,"[NVRTC] forward_dev called on unbuilt engine -- bug.\n");
            std::abort();
        }
        _zero_loss();
        void* args[]={
            (void*)&d_psi_ptr_,
            (void*)&d_params,
            (void*)&d_loss_ptr_,
            (void*)&d_xm_ptr_, (void*)&d_ym_ptr_, (void*)&d_zm_ptr_, (void*)&d_co_ptr_
        };
        CUR_CHECK(cuLaunchKernel(func_fwd_,
            1,1,1,                              // single block: SV pinned in shared mem
            blk_,1,1,
            need_optin_smem_ ? dyn_smem_bytes_ : 0,
            (CUstream)stream_, args, nullptr));
    }

    float* get_loss_ptr(){ return d_loss_.get(); }
    float2* get_psi_ptr(){ return d_psi_.get(); }
    long long n_states()const{ return n_states_; }

    ~NVRTCMegaKernelEngine(){
        if(module_) cuModuleUnload(module_);
    }

private:
    const QMLCircuit& circ_;
    const Observable& obs_;
    cudaStream_t      stream_;
    NVRTCConfig       cfg_;
    int n_;
    long long n_states_;
    int n_terms_;
    bool tiled_=false;
    bool build_ok_=true;
    bool need_optin_smem_=false;
    int  dyn_smem_bytes_=0;
    int blk_=256;

    DeviceBuffer<float2>  d_psi_;
    DeviceBuffer<float>   d_loss_;
    DeviceBuffer<unsigned> d_xm_, d_ym_, d_zm_;
    DeviceBuffer<float>    d_co_;

    // Cached pointers for arg packing (cuLaunchKernel needs pointer-to-pointer).
    float2*   d_psi_ptr_=nullptr;
    float*    d_loss_ptr_=nullptr;
    unsigned* d_xm_ptr_=nullptr;
    unsigned* d_ym_ptr_=nullptr;
    unsigned* d_zm_ptr_=nullptr;
    float*    d_co_ptr_=nullptr;

    CUmodule   module_=nullptr;
    CUfunction func_fwd_=nullptr;
    std::vector<char> ptx_;

    void _detect_cc(){
        if(cfg_.cc_major==0 && cfg_.cc_minor==0){
            cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
            cfg_.cc_major=pr.major; cfg_.cc_minor=pr.minor;
        }
        if(cfg_.cc_major==0) cfg_.cc_major=8;
    }

    void _upload_obs(){
        std::vector<unsigned> xm(n_terms_),ym(n_terms_),zm(n_terms_);
        std::vector<float> co(n_terms_);
        for(int i=0;i<n_terms_;++i){
            xm[i]=obs_.terms[i].x_mask;
            ym[i]=obs_.terms[i].y_mask;
            zm[i]=obs_.terms[i].z_mask;
            co[i]=obs_.terms[i].coeff;
        }
        d_xm_.resize(std::max(1,n_terms_));
        d_ym_.resize(std::max(1,n_terms_));
        d_zm_.resize(std::max(1,n_terms_));
        d_co_.resize(std::max(1,n_terms_));
        if(n_terms_>0){
            d_xm_.upload(xm.data(),n_terms_,stream_);
            d_ym_.upload(ym.data(),n_terms_,stream_);
            d_zm_.upload(zm.data(),n_terms_,stream_);
            d_co_.upload(co.data(),n_terms_,stream_);
        }
        d_psi_ptr_=d_psi_.get();
        d_loss_ptr_=d_loss_.get();
        d_xm_ptr_=d_xm_.get();
        d_ym_ptr_=d_ym_.get();
        d_zm_ptr_=d_zm_.get();
        d_co_ptr_=d_co_.get();
    }

    void _zero_loss(){
        CUDA_CHECK(cudaMemsetAsync(d_loss_.get(),0,sizeof(float),stream_));
    }

    // Pick block size: must be a multiple of 32, >= N_STATES/2 if possible, max 1024.
    int _choose_blk(){
        int half = (int)(n_states_>>1);
        int b = 32;
        while(b < half && b < 512) b <<= 1;
        if(b < 32) b=32;
        return b;
    }

    // ----- Codegen -----
    std::string _emit_kernel_source(){
        std::ostringstream s;
        blk_ = _choose_blk();
        const int NS = (int)n_states_;
        const int NT = n_terms_;

        s << "#define NS " << NS << "\n";
        s << "#define NT " << NT << "\n";
        s << "#define BLK " << blk_ << "\n";
        s << R"(
extern "C" {
__device__ __forceinline__ float warp_reduce(float v){
    #pragma unroll
    for(int m=16;m>0;m>>=1) v+=__shfl_xor_sync(0xffffffff,v,m);
    return v;
}
__device__ __forceinline__ float block_reduce_blk(float v){
    __shared__ float sm[32];
    int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    v=warp_reduce(v);
    if(lane==0)sm[wid]=v;
    __syncthreads();
    v=(threadIdx.x<(BLK>>5))?sm[lane]:0.f;
    if(wid==0)v=warp_reduce(v);
    return v;
}

__global__ void mega_fwd_kernel(
    float2* __restrict__ d_psi_out,
    const float* __restrict__ d_params,
    float* __restrict__ d_loss,
    const unsigned* __restrict__ x_masks,
    const unsigned* __restrict__ y_masks,
    const unsigned* __restrict__ z_masks,
    const float* __restrict__ coeffs)
{
    extern __shared__ float2 sv[];   // dynamic: host sets size at launch
    const int tid = threadIdx.x;

    for(int i=tid;i<NS;i+=BLK) sv[i] = (i==0) ? make_float2(1.f,0.f) : make_float2(0.f,0.f);
    __syncthreads();

)";

        for(size_t gi=0; gi<circ_.gates.size(); ++gi){
            s << _emit_gate(circ_.gates[gi], (int)gi);
        }

        // Expectation: sum over all i of sum_t coeff_t * <i|P_t|psi> form.
        s << R"(
    float acc = 0.f;
    for(int i=tid;i<NS;i+=BLK){
        float2 a = sv[i];
        float local = 0.f;
        #pragma unroll 1
        for(int t=0;t<NT;++t){
            unsigned xm=x_masks[t], ym=y_masks[t], zm=z_masks[t];
            int j = i ^ (int)(xm | ym);
            int zp = __popc((unsigned)i & zm) & 1;
            int yp = __popc((unsigned)i & ym) & 1;
            float sg = (zp ^ yp) ? -1.f : 1.f;
            int ny = __popc(ym);
            float2 b = sv[j];
            float c;
            switch(ny & 3){
                case 0:  c =  sg*(a.x*b.x + a.y*b.y); break;
                case 1:  c =  sg*(a.x*b.y - a.y*b.x); break;
                case 2:  c = -sg*(a.x*b.x + a.y*b.y); break;
                default: c = -sg*(a.x*b.y - a.y*b.x); break;
            }
            local += coeffs[t] * c;
        }
        acc += local;
    }
    acc = block_reduce_blk(acc);
    if(tid==0) atomicAdd(d_loss, acc);

    // Optionally publish final SV for downstream consumers.
    if(d_psi_out){
        for(int i=tid;i<NS;i+=BLK) d_psi_out[i] = sv[i];
    }
}
} // extern "C"
)";
        return s.str();
    }

    std::string _emit_gate(const QMLGate& g, int gi){
        std::ostringstream s;
        s << "    // gate " << gi << "\n";
        s << "    {\n";
        const int q  = g.qubits[0];
        const int q1 = g.qubits[1];
        const int p  = g.param_idx;
        const int qbit = (1<<q);
        switch(g.type){
            case GateType::RY:
                s << "      float th = d_params["<<p<<"];\n"
                  << "      float c, ss; __sincosf(th*0.5f,&ss,&c);\n"
                  << "      for(int t=tid; t<(NS>>1); t+=BLK){\n"
                  << "        int lo = t & ("<<(qbit-1)<<");\n"
                  << "        int hi = t >> "<<q<<";\n"
                  << "        int i0 = (hi<<"<<(q+1)<<") | lo;\n"
                  << "        int i1 = i0 | "<<qbit<<";\n"
                  << "        float2 p0=sv[i0], p1=sv[i1];\n"
                  << "        sv[i0] = make_float2(c*p0.x - ss*p1.x, c*p0.y - ss*p1.y);\n"
                  << "        sv[i1] = make_float2(ss*p0.x + c*p1.x, ss*p0.y + c*p1.y);\n"
                  << "      }\n";
                break;
            case GateType::RX:
                s << "      float th = d_params["<<p<<"];\n"
                  << "      float c, ss; __sincosf(th*0.5f,&ss,&c);\n"
                  << "      for(int t=tid; t<(NS>>1); t+=BLK){\n"
                  << "        int lo = t & ("<<(qbit-1)<<");\n"
                  << "        int hi = t >> "<<q<<";\n"
                  << "        int i0 = (hi<<"<<(q+1)<<") | lo;\n"
                  << "        int i1 = i0 | "<<qbit<<";\n"
                  << "        float2 p0=sv[i0], p1=sv[i1];\n"
                  << "        sv[i0] = make_float2(c*p0.x + ss*p1.y, c*p0.y - ss*p1.x);\n"
                  << "        sv[i1] = make_float2(ss*p0.y + c*p1.x, -ss*p0.x + c*p1.y);\n"
                  << "      }\n";
                break;
            case GateType::RZ:
                s << "      float th = d_params["<<p<<"];\n"
                  << "      float c, ss; __sincosf(th*0.5f,&ss,&c);\n"
                  << "      for(int t=tid; t<(NS>>1); t+=BLK){\n"
                  << "        int lo = t & ("<<(qbit-1)<<");\n"
                  << "        int hi = t >> "<<q<<";\n"
                  << "        int i0 = (hi<<"<<(q+1)<<") | lo;\n"
                  << "        int i1 = i0 | "<<qbit<<";\n"
                  << "        float2 p0=sv[i0], p1=sv[i1];\n"
                  << "        sv[i0] = make_float2(c*p0.x + ss*p0.y, c*p0.y - ss*p0.x);\n"
                  << "        sv[i1] = make_float2(c*p1.x - ss*p1.y, c*p1.y + ss*p1.x);\n"
                  << "      }\n";
                break;
            case GateType::H:{
                s << "      const float r = 0.70710678118f;\n"
                  << "      for(int t=tid; t<(NS>>1); t+=BLK){\n"
                  << "        int lo = t & ("<<(qbit-1)<<");\n"
                  << "        int hi = t >> "<<q<<";\n"
                  << "        int i0 = (hi<<"<<(q+1)<<") | lo;\n"
                  << "        int i1 = i0 | "<<qbit<<";\n"
                  << "        float2 p0=sv[i0], p1=sv[i1];\n"
                  << "        sv[i0] = make_float2(r*(p0.x+p1.x), r*(p0.y+p1.y));\n"
                  << "        sv[i1] = make_float2(r*(p0.x-p1.x), r*(p0.y-p1.y));\n"
                  << "      }\n";
                break;}
            case GateType::S:
                s << "      for(int i=tid; i<NS; i+=BLK){\n"
                  << "        if((i>>"<<q<<")&1){ float2 v=sv[i]; sv[i]=make_float2(-v.y, v.x); }\n"
                  << "      }\n";
                break;
            case GateType::T:
                s << "      const float r=0.70710678118f;\n"
                  << "      for(int i=tid; i<NS; i+=BLK){\n"
                  << "        if((i>>"<<q<<")&1){ float2 v=sv[i];\n"
                  << "          sv[i]=make_float2(r*v.x - r*v.y, r*v.x + r*v.y); }\n"
                  << "      }\n";
                break;
            case GateType::CNOT:{
                int qc=q, qt=q1;
                s << "      for(int i=tid; i<NS; i+=BLK){\n"
                  << "        if(!((i>>"<<qc<<")&1)) continue;\n"
                  << "        int j = i ^ "<<(1<<qt)<<";\n"
                  << "        if(j>i){ float2 t=sv[i]; sv[i]=sv[j]; sv[j]=t; }\n"
                  << "      }\n";
                break;}
            case GateType::CZ:{
                int qc=q, qt=q1;
                s << "      for(int i=tid; i<NS; i+=BLK){\n"
                  << "        if(((i>>"<<qc<<")&1)&&((i>>"<<qt<<")&1)){\n"
                  << "          float2 v=sv[i]; sv[i]=make_float2(-v.x,-v.y); }\n"
                  << "      }\n";
                break;}
            case GateType::CRZ:{
                int qc=q, qt=q1;
                s << "      float th = d_params["<<p<<"];\n"
                  << "      float c, ss; __sincosf(th*0.5f,&ss,&c);\n"
                  << "      for(int i=tid; i<NS; i+=BLK){\n"
                  << "        if(!((i>>"<<qc<<")&1)) continue;\n"
                  << "        float2 v=sv[i];\n"
                  << "        float sgn=((i>>"<<qt<<")&1)?+1.f:-1.f;\n"
                  << "        sv[i]=make_float2(c*v.x - sgn*ss*v.y, c*v.y + sgn*ss*v.x);\n"
                  << "      }\n";
                break;}
            default:
                s << "      /* barrier */\n";
                break;
        }
        s << "      __syncthreads();\n";
        s << "    }\n";
        return s.str();
    }

    void _build_and_load(){
        std::string src = _emit_kernel_source();
        if(cfg_.keep_ptx){
            fprintf(stderr,"[NVRTC] generated %zu bytes of CUDA source\n", src.size());
        }
        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "qhpc_mega.cu", 0, nullptr, nullptr));

        std::string arch = "--gpu-architecture=sm_"
            + std::to_string(cfg_.cc_major) + std::to_string(cfg_.cc_minor);
        std::vector<const char*> opts;
        opts.push_back(arch.c_str());
        // C++20 to stay in sync with the rest of the core (concepts, type_traits...).
        opts.push_back("-std=c++20");
        if(cfg_.use_fast_math) opts.push_back("--use_fast_math");
        opts.push_back("-default-device");

        nvrtcResult cr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
        if(cr != NVRTC_SUCCESS){
            size_t logSize=0; nvrtcGetProgramLogSize(prog,&logSize);
            std::vector<char> log(logSize+1,0);
            nvrtcGetProgramLog(prog,log.data());
            fprintf(stderr,"[NVRTC] compile failed:\n%s\n", log.data());
            nvrtcDestroyProgram(&prog);
            // Do NOT abort: trainer falls back to STATEVEC_KERNELS.
            func_fwd_ = nullptr;
            return;
        }
        size_t ptxSize=0; NVRTC_CHECK(nvrtcGetPTXSize(prog,&ptxSize));
        ptx_.assign(ptxSize,0);
        NVRTC_CHECK(nvrtcGetPTX(prog, ptx_.data()));
        NVRTC_CHECK(nvrtcDestroyProgram(&prog));

        // Driver API load.
        CUR_CHECK(cuInit(0));
        CUR_CHECK(cuModuleLoadDataEx(&module_, ptx_.data(), 0, nullptr, nullptr));
        CUR_CHECK(cuModuleGetFunction(&func_fwd_, module_, "mega_fwd_kernel"));

        // Opt in to >48KB dynamic shared memory on Volta+ when needed.
        if(need_optin_smem_){
            CUR_CHECK(cuFuncSetAttribute(func_fwd_,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dyn_smem_bytes_));
        }
    }
};
