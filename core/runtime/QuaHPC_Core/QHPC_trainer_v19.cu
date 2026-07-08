#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include "QHPC_quantum_ir_v19.hpp"
#include "QHPC_selector_v19.hpp"
#include "QHPC_adjoint_diff_v19.cu"
#include "QHPC_nvrtc_megakernel_v19.cu"
#include "QHPC_optimizer_v19.cu"
#include "QHPC_qng_v19.hpp"
#include "QHPC_tpb_v19.hpp"
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <memory>
#include <atomic>

using hrc=std::chrono::high_resolution_clock;
using fms=std::chrono::duration<float,std::milli>;

__global__ __launch_bounds__(256)
void param_init_rand_f32(float*__restrict__ p,int n,unsigned seed){
    int i=blockIdx.x*256+threadIdx.x;if(i>=n)return;
    unsigned s=seed^(unsigned)i*2654435761u;
    s^=s>>16;s*=0x45d9f3bu;s^=s>>16;
    p[i]=((float)(s&0xffffffu)/(float)0xffffffu-0.5f)*0.2f;
}

// Append a loss value into a device-resident ring buffer. Host samples it
// every N epochs, avoiding a per-step PCIe transfer.
//
// CORRECTNESS:
//  - Slot must be flushed across PCIe BEFORE the index is published, else
//    the host can read the new index but still see stale loss data in cache.
//  - Order:  (a) read 'next' slot via reserve atomic, (b) write payload,
//            (c) __threadfence_system() to force PCIe flush of the payload,
//            (d) bump the published counter.
//  - We use two counters: ring_idx_reserve_ (next free slot, incremented
//    atomically BEFORE write), and ring_idx_pub_ (published count, bumped
//    atomically AFTER fence). The host reads ring_idx_pub_ — guaranteeing
//    every slot below it has been flushed.
__global__ __launch_bounds__(1)
void append_loss_ring(float* ring, int ring_size,
                       int* ring_idx_reserve, int* ring_idx_pub,
                       const float* loss_src){
    int slot = atomicAdd(ring_idx_reserve, 1) % ring_size;
    ring[slot] = *loss_src;
    // System-level fence: ensure the write to `ring[slot]` is visible to the
    // CPU across the PCIe bus before we publish the new index.
    __threadfence_system();
    atomicAdd(ring_idx_pub, 1);
    // A second fence so the index update is also flushed promptly. Without
    // this, the host could spin on the prior index for a while.
    __threadfence_system();
}

// QNG-SGD step: simple θ ← θ − η Δ used in place of Adam when QNG is on.
// Mixing Adam's RMS-based moment rescaling with QNG's curvature-corrected
// update direction causes the optimiser to oscillate and never converge,
// so the QNG path bypasses Adam entirely.
__global__ __launch_bounds__(256)
void qng_apply_sgd_step(float* params, const float* delta, float lr, int P){
    int i = blockIdx.x * 256 + threadIdx.x;
    if(i >= P) return;
    params[i] -= lr * delta[i];
}

class QMLTrainer {
public:
    QMLTrainer(QMLCircuit circ, Observable obs, QMLConfig cfg,
               cudaStream_t ext_stream=nullptr)
        : raw_circ_(std::move(circ)), obs_(std::move(obs)), cfg_(cfg){
        cudaStreamCreate(&compute_);
        if(ext_stream) compute_=ext_stream;

        ir_ = QuantumIR::from_circuit(raw_circ_);
        run_all_passes(ir_, cfg_.verbose_ir);
        opt_circ_ = ir_.to_circuit();

        BackendPick bp = pick_backend(ir_, obs_, cfg_);
        if(cfg_.verbose_selector) log_backend_pick(bp);
        active_backend_ = bp.kind;

        int P = ir_.n_params;
        d_params_.resize(P);
        d_grads_.resize(P);
        h_params_inspect_.resize(P, 0.f);

        ring_size_ = std::max(64, cfg_.n_epochs);
        d_loss_ring_.resize(ring_size_);
        d_ring_idx_reserve_.resize(1);
        d_ring_idx_pub_.resize(1);
        cudaMemsetAsync(d_ring_idx_reserve_.get(), 0, sizeof(int), compute_);
        cudaMemsetAsync(d_ring_idx_pub_.get(),     0, sizeof(int), compute_);

        // Pinned host mirror for ring contents (one DMA per drain).
        CUDA_CHECK(cudaHostAlloc(&h_loss_ring_, ring_size_*sizeof(float),
            cudaHostAllocPortable));

        _init_params();

        adam_ = std::make_unique<AdamGPU>(P, cfg_.lr, cfg_.beta1, cfg_.beta2,
                                          cfg_.eps_adam, compute_, cfg_.adamw);
        scaler_ = std::make_unique<GradScaler>(P, cfg_.grad_scaler, compute_);
        sched_  = std::make_unique<LRScheduler>(cfg_.lr, cfg_.n_epochs,
            LRScheduler::Mode::COSINE, cfg_.lr*.01f, 10);

        _build_active_backend();

        // v17: Quantum Natural Gradient.  Construct iff requested + we have
        // a SV-style engine that exposes psi/mu_pn buffers.  P*N memory cost
        // is checked: skip silently if it would exceed 2 GB.
        if(cfg_.use_qng && active_backend_ == BackendKind::STATEVEC_KERNELS){
            long long N = 1LL << ir_.n_qubits;
            size_t mu_bytes = (size_t)P * (size_t)N * sizeof(float2);
            if(mu_bytes > (size_t)2 << 30){
                fprintf(stderr,
                    "[QNG] DISABLED — μ buffer would need %.2f GB (P=%d, N=%lld). "
                    "Lower n_qubits or n_params, or disable use_qng.\n",
                    (double)mu_bytes / (1024.0*1024.0*1024.0), P, N);
            } else {
                qng_ = std::make_unique<QNGEngine>(P, N, compute_);
            }
        }
        _print_banner();
    }

    ~QMLTrainer(){
        for(auto& kv:graph_exec_cache_) if(kv.second) cudaGraphExecDestroy(kv.second);
        if(graph_) cudaGraphDestroy(graph_);
        cudaStreamDestroy(compute_);
        if(h_loss_ring_) cudaFreeHost(h_loss_ring_);
    }

    QMLMetrics train(){
        QMLMetrics met;
        auto t0=hrc::now();
        for(int ep=0; ep<cfg_.n_epochs; ++ep){
            adam_->set_lr(sched_->get(ep));
            _step(ep);
            // Periodic drain: pull ring → host every print_every (or 50) epochs.
            int drain_every = std::max(50, cfg_.print_every);
            if((ep+1) % drain_every == 0 || ep==cfg_.n_epochs-1){
                _drain_loss_ring(met);
                if(cfg_.print_every>0){
                    float L = met.loss_history.empty()?0.f:met.loss_history.back();
                    fprintf(stderr,"  [%4d/%4d]  loss=%-10.6f  lr=%.5f\n",
                        ep, cfg_.n_epochs, L, sched_->get(ep));
                }
                if(_converged(met)) break;
            }
        }
        _drain_loss_ring(met);
        CUDA_CHECK(cudaStreamSynchronize(compute_));

        CUDA_CHECK(cudaMemcpyAsync(h_params_inspect_.data(), d_params_.get(),
            ir_.n_params*sizeof(float), cudaMemcpyDeviceToHost, compute_));
        CUDA_CHECK(cudaStreamSynchronize(compute_));

        float elapsed = fms(hrc::now()-t0).count()*0.001f;
        met.wall_time_s = elapsed;
        met.throughput_keps = (float)met.loss_history.size()/elapsed/1000.f;
        for(size_t i=0;i<met.loss_history.size();++i){
            if(met.loss_history[i] < met.best_loss){
                met.best_loss = met.loss_history[i];
                met.best_epoch = (int)i;
            }
        }
        fprintf(stderr,
            "+==========================================+\n"
            "|  Done: best=%-10.6f  ep=%-4d         |\n"
            "|  Time: %-8.2fs  Tput: %-8.3f kEp/s |\n"
            "+==========================================+\n",
            met.best_loss, met.best_epoch, met.wall_time_s, met.throughput_keps);
        return met;
    }

    std::vector<float> get_params(){
        CUDA_CHECK(cudaMemcpyAsync(h_params_inspect_.data(), d_params_.get(),
            ir_.n_params*sizeof(float), cudaMemcpyDeviceToHost, compute_));
        CUDA_CHECK(cudaStreamSynchronize(compute_));
        return h_params_inspect_;
    }
    void set_params(const std::vector<float>& p){
        assert((int)p.size() == ir_.n_params);
        CUDA_CHECK(cudaMemcpyAsync(d_params_.get(), p.data(),
            p.size()*sizeof(float), cudaMemcpyHostToDevice, compute_));
        CUDA_CHECK(cudaStreamSynchronize(compute_));
    }
    void set_lr(float lr){cfg_.lr=lr; adam_->set_lr(lr);}
    BackendKind active_backend()const{return active_backend_;}

private:
    QMLCircuit  raw_circ_;
    QuantumIR   ir_;
    QMLCircuit  opt_circ_;
    Observable  obs_;
    QMLConfig   cfg_;
    BackendKind active_backend_;

    cudaStream_t compute_=nullptr;
    cudaGraph_t  graph_=nullptr;
    std::vector<std::pair<int,cudaGraphExec_t>> graph_exec_cache_;

    int ring_size_=0;
    int ring_drained_=0;
    DeviceBuffer<float> d_loss_ring_;
    DeviceBuffer<int>   d_ring_idx_reserve_;   // next free slot (write-side counter)
    DeviceBuffer<int>   d_ring_idx_pub_;       // committed count (read by host)
    float* h_loss_ring_ = nullptr;

    DeviceBuffer<float> d_params_, d_grads_;
    std::vector<float>  h_params_inspect_;

    std::unique_ptr<TrueAdjointDiffEngine>  sv_engine_;
    std::unique_ptr<NVRTCMegaKernelEngine>  nvrtc_engine_;

    std::unique_ptr<AdamGPU>     adam_;
    std::unique_ptr<GradScaler>  scaler_;
    std::unique_ptr<LRScheduler> sched_;
    std::unique_ptr<QNGEngine>   qng_;

    void _init_params(){
        unsigned seed=(unsigned)std::chrono::steady_clock::now().time_since_epoch().count()&0xffffffffu;
        int P=ir_.n_params, bl=(P+255)/256;
        param_init_rand_f32<<<bl,256,0,compute_>>>(d_params_.get(),P,seed);
        CUDA_CHECK(cudaStreamSynchronize(compute_));
    }

    void _build_active_backend(){
        switch(active_backend_){
            case BackendKind::STATEVEC_KERNELS:
                sv_engine_ = std::make_unique<TrueAdjointDiffEngine>(
                    opt_circ_, obs_, compute_, cfg_.fused_kernels, cfg_.use_tpb);
                break;
            case BackendKind::NVRTC_MEGAKERNEL:
                nvrtc_engine_ = std::make_unique<NVRTCMegaKernelEngine>(
                    opt_circ_, obs_, compute_, cfg_.nvrtc);
                if(!nvrtc_engine_->ok()){
                    fprintf(stderr,"[Trainer] NVRTC failed -> STATEVEC fallback.\n");
                    active_backend_ = BackendKind::STATEVEC_KERNELS;
                    nvrtc_engine_.reset();
                    sv_engine_ = std::make_unique<TrueAdjointDiffEngine>(
                        opt_circ_, obs_, compute_, cfg_.fused_kernels, cfg_.use_tpb);
                }
                break;
            default:
                fprintf(stderr,"[Trainer] Backend %d not adjoint-capable in this build "
                    "-> STATEVEC fallback for gradients.\n", (int)active_backend_);
                sv_engine_ = std::make_unique<TrueAdjointDiffEngine>(
                    opt_circ_, obs_, compute_, cfg_.fused_kernels, cfg_.use_tpb);
                active_backend_ = BackendKind::STATEVEC_KERNELS;
                break;
        }
    }

    void _print_banner(){
        const char* bn="?";
        switch(active_backend_){
            case BackendKind::STATEVEC_KERNELS: bn="STATEVEC"; break;
            case BackendKind::NVRTC_MEGAKERNEL: bn="NVRTC-MK"; break;
            case BackendKind::HEISENBERG:       bn="HEISEN";   break;
            case BackendKind::MPS:              bn="MPS";      break;
            case BackendKind::DISTRIBUTED:      bn="DIST";     break;
            default: break;
        }
        const char* opt_name = (cfg_.optimizer==OptimizerKind::ADAMW)?"AdamW":"Adam";
        fprintf(stderr,
            "\n+======================================================+\n"
            "|   QHPC-QML v19.1 Graph-safe + QNG-SGD + Chain1q      |\n"
            "+------------------------------------------------------+\n"
            "|  Qubits : %-4d  Params : %-4d  IR-gates : %-6zu    |\n"
            "|  Backend: %-8s  Optim : %-6s  Scaler : %-3s       |\n"
            "|  Graph  : %-3s    Loss-ring : %-4d  Grad : atomic-free |\n"
            "|  QNG : %-3s   TPB : %-3s   damping=%-6.4f             |\n"
            "|  cp.async double-buffer + 3-MMA complex TC + cuSOLVER ws |\n"
            "|  Epochs : %-6d  LR    : %-10.5f                  |\n"
            "+======================================================+\n",
            ir_.n_qubits, ir_.n_params,
            std::count_if(ir_.insts.begin(),ir_.insts.end(),
                [](const IRInst& i){return !i.dead && i.kind!=IRKind::NOP;}),
            bn, opt_name, cfg_.grad_scaler.enabled?"YES":"NO",
            cfg_.use_cuda_graph?"YES":"NO", ring_size_,
            cfg_.use_qng?"YES":"NO", cfg_.use_tpb?"YES":"NO", cfg_.qng_damping,
            cfg_.n_epochs, cfg_.lr);
    }

    void _step(int epoch){
        if(cfg_.use_cuda_graph && active_backend_==BackendKind::STATEVEC_KERNELS){
            _graph_step_sv(epoch);
            return;
        }
        _eager_step();
    }

    void _eager_step(){
        sv_engine_->forward_backward_dev(d_params_.get(), d_grads_.get());

        // v17: Quantum Natural Gradient transformation. Replace ∇L by g⁻¹·∇L
        // where g is the Fubini-Study metric. Skip when use_qng is off, or
        // when the QNGEngine could not be allocated (P too large, etc).
        if(cfg_.use_qng && qng_ && qng_->ready()){
            // Materialise per-parameter ∂_i ψ vectors via parameter shift.
            sv_engine_->materialise_param_derivatives(
                d_params_.get(),
                qng_->mu_row(0),
                qng_->n_states());
            // Need a fresh forward to put ψ(θ) into d_psi_ for the projector term.
            sv_engine_->forward_dev(d_params_.get());
            qng_->contract_metric(sv_engine_->get_psi_ptr());
            qng_->natural_grad_inplace(d_grads_.get(), cfg_.qng_damping);

            // IMPORTANT: QNG produces a metric-corrected direction Δ = g⁻¹∇L
            // that is fundamentally different in scale and direction from a
            // raw ∇L. Adam's adaptive moments (m, v) calibrated on raw
            // gradients become POISONED when fed a g⁻¹∇L stream — the
            // optimiser oscillates between QNG's curvature-aware step and
            // Adam's RMS-based rescaling, and convergence stalls (this is
            // the loss ≈ 0.17 / never-decreasing pattern observed in v19).
            //
            // The literature is consistent: QNG is used with PLAIN SGD,
            // i.e. θ ← θ − η Δ.  Apply that directly here, bypassing Adam.
            qng_apply_sgd_step<<<(ir_.n_params+255)/256, 256, 0, compute_>>>(
                d_params_.get(), d_grads_.get(), adam_->lr(), ir_.n_params);
            // After QNG-SGD step, still need to publish the loss for logging.
            append_loss_ring<<<1,1,0,compute_>>>(
                d_loss_ring_.get(), ring_size_,
                d_ring_idx_reserve_.get(), d_ring_idx_pub_.get(),
                sv_engine_->get_loss_ptr());
            return;
        }

        const float* scale_ptr = scaler_->enabled() ? scaler_->scale_ptr() : nullptr;
        if(cfg_.grad_clip_norm>0.f || cfg_.adamw.enabled || scale_ptr){
            adam_->clip_and_step(d_params_.get(), d_grads_.get(),
                                 cfg_.grad_clip_norm, scale_ptr);
        } else {
            adam_->step(d_params_.get(), d_grads_.get());
        }
        append_loss_ring<<<1,1,0,compute_>>>(
            d_loss_ring_.get(), ring_size_, d_ring_idx_reserve_.get(), d_ring_idx_pub_.get(),
            sv_engine_->get_loss_ptr());
    }

    int _lr_key()const{ return (int)(adam_->lr()*1e6f); }

    void _graph_step_sv(int /*epoch*/){
        int key = _lr_key();
        cudaGraphExec_t exec=nullptr;
        for(auto& kv:graph_exec_cache_) if(kv.first==key){exec=kv.second; break;}
        if(!exec){
            exec = _capture_sv_step();
            if(exec) graph_exec_cache_.push_back({key,exec});
        }
        if(!exec){
            cfg_.use_cuda_graph=false;
            _eager_step();
            return;
        }
        CUDA_CHECK(cudaGraphLaunch(exec, compute_));
    }

    cudaGraphExec_t _capture_sv_step(){
        CUDA_CHECK(cudaStreamSynchronize(compute_));
        cudaGraph_t g=nullptr;
        if(cudaStreamBeginCapture(compute_, cudaStreamCaptureModeThreadLocal)!=cudaSuccess)
            return nullptr;
        sv_engine_->forward_backward_dev(d_params_.get(), d_grads_.get());
        const float* scale_ptr = scaler_->enabled() ? scaler_->scale_ptr() : nullptr;
        adam_->clip_and_step(d_params_.get(), d_grads_.get(),
                             cfg_.grad_clip_norm, scale_ptr);
        append_loss_ring<<<1,1,0,compute_>>>(
            d_loss_ring_.get(), ring_size_, d_ring_idx_reserve_.get(), d_ring_idx_pub_.get(),
            sv_engine_->get_loss_ptr());
        if(cudaStreamEndCapture(compute_, &g)!=cudaSuccess) return nullptr;
        cudaGraphExec_t exec=nullptr;
        if(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0)!=cudaSuccess){
            cudaGraphDestroy(g); return nullptr;
        }
        if(graph_) cudaGraphDestroy(graph_);
        graph_=g;
        return exec;
    }

    void _drain_loss_ring(QMLMetrics& met){
        // Read the PUBLISHED counter. Every kernel that bumped this counter
        // already issued __threadfence_system() AFTER its loss-slot write,
        // so by the time we observe cur_idx == N, slots [0..N) are flushed.
        int cur_idx = 0;
        CUDA_CHECK(cudaMemcpyAsync(&cur_idx, d_ring_idx_pub_.get(), sizeof(int),
            cudaMemcpyDeviceToHost, compute_));
        CUDA_CHECK(cudaMemcpyAsync(h_loss_ring_, d_loss_ring_.get(),
            ring_size_ * sizeof(float), cudaMemcpyDeviceToHost, compute_));
        CUDA_CHECK(cudaStreamSynchronize(compute_));
        // Acquire fence on the host so the compiler / CPU can't hoist the
        // reads of h_loss_ring_ above the cur_idx observation.
        std::atomic_thread_fence(std::memory_order_acquire);

        int n_new = cur_idx - ring_drained_;
        if(n_new > ring_size_){
            fprintf(stderr,"[Trainer] loss ring overflow: drained=%d cur=%d size=%d\n",
                ring_drained_, cur_idx, ring_size_);
            n_new = ring_size_;
        }
        for(int k=0;k<n_new;++k){
            int slot = (ring_drained_ + k) % ring_size_;
            met.loss_history.push_back(h_loss_ring_[slot]);
        }
        ring_drained_ = cur_idx;
    }

    bool _converged(const QMLMetrics& m){
        int n=(int)m.loss_history.size();
        int w=cfg_.convergence_win;
        if(n<2*w) return false;
        float a=0.f,b=0.f;
        for(int i=n-w;i<n;++i) a+=m.loss_history[i];
        for(int i=n-2*w;i<n-w;++i) b+=m.loss_history[i];
        return std::abs(a-b)/(w+1e-9f) < cfg_.convergence_tol;
    }
};

inline QMLMetrics run_vqe(int n,int layers,const Observable& H,QMLConfig cfg={}){
    auto circ=QMLCircuit::hardware_efficient(n,layers);
    return QMLTrainer(std::move(circ),H,cfg).train();
}
inline QMLMetrics run_qaoa(int n,int p_layers,
    const std::vector<std::pair<int,int>>& E,QMLConfig cfg={}){
    auto circ=QMLCircuit::qaoa(n,p_layers,E);
    auto obs =Observable::max_cut(n,E);
    return QMLTrainer(std::move(circ),std::move(obs),cfg).train();
}
inline QMLMetrics run_qml_classifier(int n,int layers,const Observable& obs,QMLConfig cfg={}){
    cfg.use_cuda_graph=true;
    auto circ=QMLCircuit::hardware_efficient(n,layers);
    return QMLTrainer(std::move(circ),obs,cfg).train();
}
