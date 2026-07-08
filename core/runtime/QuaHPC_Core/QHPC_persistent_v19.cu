#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include <chrono>

// ============================================================================
// Persistent kernel runtime — v15.
// ----------------------------------------------------------------------------
// FIXES vs v13:
//
// 1) PCIe flooding: v13 had __nanosleep(50) inside the spin loop, which on
//    Ada/Hopper still hits PCIe several million times per second when slot_
//    is allocated via cudaHostAlloc | cudaHostAllocMapped (zero-copy). v14
//    uses __nanosleep(spin_backoff_ns) with a default of 1000 ns (1 μs),
//    knocking poll rate to ~1 MHz and freeing PCIe for the actual command.
//
// 2) Memory ordering: writes to slot->dev_seq are preceded by
//    __threadfence_system() so the CPU sees them in order across the PCIe
//    bus, not stuck in some GPU write coalescer.
//
// 3) OS watchdog: the kernel breaks the loop after watchdog_break_ms with no
//    command. Host can re-launch on next step. Avoids Windows TDR / Linux Xid
//    if the user pauses the trainer for >2 s.
//
// 4) Volatile reads: all polled fields are volatile-qualified in the struct,
//    so the compiler cannot hoist or cache them in registers.
//
// PERFORMANCE PROFILE:
//   - PCIe poll rate:   ~1 MHz   (was ~20 MHz on v13)
//   - Per-step overhead: ~2-5 μs (was ~10-20 μs with regular launch)
//   - Best fit:          shallow circuits (n<=10), epoch time < 100 μs
// ============================================================================

struct PersistentSlot {
    volatile unsigned long long host_seq;
    volatile unsigned long long dev_seq;
    volatile float                last_loss;
    volatile int                  shutdown;
};

// JIT-emitted by NVRTC at runtime; signature stable for linkage.
extern "C" __device__ void qhpc_persistent_step(
    const float* d_params,
    float* d_grads,
    float* d_loss_out);

__global__ void qhpc_persistent_runtime(
    PersistentSlot* __restrict__ slot,
    const float* __restrict__ d_params,
    float* __restrict__ d_grads,
    float* __restrict__ d_loss_out,
    int spin_backoff_ns,
    unsigned long long watchdog_iters)
{
    if(blockIdx.x != 0) return;
    unsigned long long last_seen = 0;
    unsigned long long idle_iters = 0;
    while(true){
        unsigned long long s = slot->host_seq;
        if(slot->shutdown) break;
        if(s == last_seen){
            // Backoff: 1 μs default. Reduces PCIe poll bandwidth by ~20x vs v13.
            __nanosleep(spin_backoff_ns);
            ++idle_iters;
            // Watchdog: if we've been idle for too long, exit gracefully so
            // the OS doesn't kill us. Host can restart the dispatcher.
            if(idle_iters > watchdog_iters) break;
            continue;
        }
        idle_iters = 0;
        last_seen = s;
        qhpc_persistent_step(d_params, d_grads, d_loss_out);
        // Make the loss write visible to the CPU before dev_seq update.
        __threadfence_system();
        slot->last_loss = *d_loss_out;
        __threadfence_system();
        slot->dev_seq = s;
    }
}

class PersistentRuntime {
public:
    PersistentRuntime(cudaStream_t stream, int n_params,
                       const PersistentConfig& cfg = PersistentConfig{})
        : stream_(stream), n_params_(n_params), cfg_(cfg){
        CUDA_CHECK(cudaHostAlloc(&slot_, sizeof(PersistentSlot),
            cudaHostAllocPortable | cudaHostAllocMapped));
        slot_->host_seq=0; slot_->dev_seq=0; slot_->shutdown=0; slot_->last_loss=0.f;
        cudaHostGetDevicePointer((void**)&d_slot_, slot_, 0);
        d_grads_.resize(n_params);
        d_loss_.resize(1);
        // ~1 MHz poll * 1500 ms = 1.5e6 idle iters before watchdog break.
        watchdog_iters_ = (unsigned long long)(
            (double)cfg_.watchdog_break_ms * 1e6 /
            std::max(1, cfg_.spin_backoff_ns));
    }
    ~PersistentRuntime(){
        if(running_){
            slot_->shutdown = 1;
            __sync_synchronize();
            cudaStreamSynchronize(stream_);
        }
        if(slot_) cudaFreeHost(slot_);
    }

    void start(const float* d_params){
        qhpc_persistent_runtime<<<1, 32, 0, stream_>>>(
            d_slot_, d_params, d_grads_.get(), d_loss_.get(),
            cfg_.spin_backoff_ns, watchdog_iters_);
        running_ = true;
    }

    // Returns the loss of the triggered step. Times out gracefully if the
    // device-side kernel hit its watchdog.
    float trigger_step(){
        unsigned long long s = ++step_seq_;
        slot_->host_seq = s;
        __sync_synchronize();
        auto t0 = std::chrono::steady_clock::now();
        while(slot_->dev_seq != s){
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if(dt > cfg_.watchdog_break_ms + 500){
                // Device watchdog must have fired; relaunch.
                running_ = false;
                return std::numeric_limits<float>::quiet_NaN();
            }
            // CPU side yields to avoid burning a core.
            std::this_thread::yield();
        }
        return slot_->last_loss;
    }

    void stop(){
        if(!running_) return;
        slot_->shutdown = 1;
        cudaStreamSynchronize(stream_);
        running_ = false;
    }

    float* grads_ptr(){ return d_grads_.get(); }
    float* loss_ptr() { return d_loss_.get(); }
    bool   alive()    { return running_; }

private:
    cudaStream_t stream_;
    int n_params_;
    PersistentConfig cfg_;
    PersistentSlot* slot_=nullptr;
    PersistentSlot* d_slot_=nullptr;
    DeviceBuffer<float> d_grads_;
    DeviceBuffer<float> d_loss_;
    bool  running_=false;
    unsigned long long step_seq_=0;
    unsigned long long watchdog_iters_=0;
};
