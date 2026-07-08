#pragma once
// ============================================================================
// QHPC_qng_v19.hpp  —  Quantum Natural Gradient (Fubini-Study metric)
// ----------------------------------------------------------------------------
// Replaces the flat Euclidean gradient step with the Riemannian step
//
//     Δθ = g⁻¹ · ∇L
//
// where g_ij is the Fubini-Study metric on parameter space:
//
//     g_ij = Re(⟨∂_i ψ | ∂_j ψ⟩) − ⟨∂_i ψ | ψ⟩ ⟨ψ | ∂_j ψ⟩
//
// Why this matters in QML
// -----------------------
// Adam/L-BFGS were designed for flat Euclidean DL parameter spaces.  Quantum
// parameter space is a *Riemannian manifold* — distances are warped, and
// gradient flows get stuck in plateaus (the barren-plateau pathology) when
// the steepest direction in θ-coordinates is NOT the steepest direction on
// the state manifold.  The Fubini-Study metric tells us the *real*
// infinitesimal distance between |ψ(θ)⟩ and |ψ(θ+dθ)⟩.  Stepping in metric-
// corrected direction
//
//     θ ← θ − η g⁻¹ ∇L
//
// is to QML what Newton's method is to convex optimization: a couple of
// epochs often replace hundreds of vanilla-Adam epochs.
//
// Implementation
// --------------
//  * During the adjoint backward pass we already materialise
//    |∂_i ψ⟩  for each parameter i (this is what the fused-grad kernel
//    computes locally).  We persist these vectors into a P × N_states
//    buffer  d_mu_  (P = n_params, N_states = 2^n).
//  * Fubini-Study metric:
//        A_ij = ⟨∂_i ψ | ∂_j ψ⟩      = (d_mu)ᴴ (d_mu)        [cuBLAS Cgemm]
//        b_i  = ⟨∂_i ψ | ψ⟩          = (d_mu)ᴴ ψ              [cuBLAS Cgemv]
//        g_ij = Re(A_ij) − Re(b_i) Re(b_j) − Im(b_i) Im(b_j)
//      (= Re(A_ij) − Re(b_i* · b_j) when expanded)
//  * Damp the diagonal by  λ I  (Tikhonov) to keep the solve well-conditioned
//    AND to make g strictly Symmetric Positive Definite (SPD).
//  * Solve  g Δ = ∇L  with **Cholesky** factorisation (cuSOLVER potrf/potrs).
//    The FIM is SPD by construction (it's a Gram matrix of state derivatives
//    plus a positive diagonal), so Cholesky is the right choice:
//      - half the flops of LU  (P³/3 vs 2P³/3)
//      - no pivoting array     (saves P * sizeof(int) VRAM)
//      - more numerically stable for ill-conditioned metrics
//  * Apply  θ ← θ − η Δ.
//
// Workspace is pre-allocated ONCE at engine construction (same discipline as
// the MPS engine's cuSOLVER buffer).  No malloc on the hot path.
// ============================================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>
#include "QHPC_memory_v10.hpp"

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(x) do { cublasStatus_t _s = (x); \
    if(_s != CUBLAS_STATUS_SUCCESS){ \
        fprintf(stderr,"[cuBLAS] error %d at %s:%d\n",(int)_s,__FILE__,__LINE__); \
        std::abort(); } } while(0)
#endif
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(x) do { cusolverStatus_t _s = (x); \
    if(_s != CUSOLVER_STATUS_SUCCESS){ \
        fprintf(stderr,"[cuSOLVER] error %d at %s:%d\n",(int)_s,__FILE__,__LINE__); \
        std::abort(); } } while(0)
#endif

// ---------------------------------------------------------------------------
// FIM assembly kernel: real Fubini-Study metric from complex A_ij and b_i.
// One thread per (i, j) entry of the P × P metric.  We expand
//     g_ij = Re(A_ij) − Re(b_i*) Re(b_j) + Im(b_i*) Im(b_j)
//          = Re(A_ij) − Re(b_i) Re(b_j) − Im(b_i) Im(b_j)
// then add Tikhonov damping  λ  on the diagonal.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void qng_assemble_fim_kernel(
    const float2* __restrict__ A,   // P*P complex matrix (column-major), <∂iψ|∂jψ>
    const float2* __restrict__ b,   // P complex vector,                    <∂iψ|ψ>
    float*        __restrict__ g,   // P*P real metric (column-major), output
    int    P,
    float  damping)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= P || j >= P) return;
    float2 a  = A[(size_t)j * P + i];
    float2 bi = b[i];
    float2 bj = b[j];
    float val = a.x - (bi.x * bj.x + bi.y * bj.y);
    if(i == j) val += damping;
    g[(size_t)j * P + i] = val;
}

// ---------------------------------------------------------------------------
// QNGEngine — owns FIM workspace, cuBLAS handle, cuSOLVER handle.
// Lifetime tied to the trainer.  Construct once; .step() per epoch.
// ---------------------------------------------------------------------------
class QNGEngine {
public:
    QNGEngine(int n_params, long long n_states, cudaStream_t stream=nullptr)
        : P_(n_params), N_(n_states), stream_(stream)
    {
        if(P_ <= 0 || N_ <= 0){ ready_ = false; return; }
        // Per-parameter derivative vectors |∂_i ψ⟩ — P × N complex.
        d_mu_.resize((size_t)P_ * (size_t)N_);
        d_A_ .resize((size_t)P_ * (size_t)P_);
        d_b_ .resize((size_t)P_);
        d_g_ .resize((size_t)P_ * (size_t)P_);
        d_rhs_.resize((size_t)P_);
        d_info_.resize(1);
        // NOTE: no d_ipiv_ needed — Cholesky doesn't pivot.

        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver_, stream_));

        // FIM is SYMMETRIC POSITIVE DEFINITE once Tikhonov damping is added
        // to the diagonal. That makes Cholesky the right factorisation:
        //   * ~2× fewer flops than LU (P³/3 vs 2P³/3)
        //   * numerically stable WITHOUT pivoting (so no ipiv buffer)
        //   * exploits symmetry, halves the working set
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
            cusolver_, CUBLAS_FILL_MODE_LOWER, P_,
            d_g_.get(), P_, &lwork));
        d_work_.resize(std::max(1, lwork));
        lwork_ = lwork;

        fprintf(stderr,
            "[QNG] init: P=%d  N=%lld  mu-buf=%.2f MB  FIM=%.2f MB  "
            "solver-ws=%.2f MB  (Cholesky)\n",
            P_, N_,
            (double)P_ * N_ * sizeof(float2) / (1024.0*1024.0),
            (double)P_ * P_ * sizeof(float)  / (1024.0*1024.0),
            (double)lwork * sizeof(float)    / (1024.0*1024.0));
        ready_ = true;
    }

    ~QNGEngine(){
        if(cublas_)   cublasDestroy(cublas_);
        if(cusolver_) cusolverDnDestroy(cusolver_);
    }

    bool ready() const { return ready_; }
    float2* mu_row(int i){ return d_mu_.get() + (size_t)i * N_; }
    long long n_states() const { return N_; }
    int n_params() const { return P_; }

    // Compute A = mu^H mu  (P x P), b = mu^H ψ  (P).
    void contract_metric(const float2* d_psi){
        float2 alpha = {1.f, 0.f};
        float2 beta  = {0.f, 0.f};
        // A = mu^H * mu : (P x N) x (N x P) -> (P x P)
        //   In column-major view, mu is stored as P rows of N (row-major in
        //   memory). We treat it as an (N x P) col-major matrix and do
        //   A = mu^H * mu using cublasCgemm with op=Conj-Transpose on A.
        CUBLAS_CHECK(cublasCgemm(
            cublas_, CUBLAS_OP_C, CUBLAS_OP_N,
            P_, P_, (int)N_,
            &alpha,
            d_mu_.get(),  (int)N_,
            d_mu_.get(),  (int)N_,
            &beta,
            d_A_.get(),   P_));
        // b = mu^H * ψ : (P x N) x (N x 1) -> (P x 1)
        CUBLAS_CHECK(cublasCgemv(
            cublas_, CUBLAS_OP_C,
            (int)N_, P_,
            &alpha,
            d_mu_.get(),  (int)N_,
            d_psi,        1,
            &beta,
            d_b_.get(),   1));
    }

    // Assemble real FIM with Tikhonov damping, then solve g Δ = ∇L in-place
    // using Cholesky (FIM is SPD). On return d_grad holds Δ.
    void natural_grad_inplace(float* d_grad, float damping = 1e-3f){
        // 1) FIM = Re(A) − Re(b) Re(b) − Im(b) Im(b)  + λ I
        dim3 blk(16, 16);
        dim3 grd((P_ + 15) / 16, (P_ + 15) / 16);
        qng_assemble_fim_kernel<<<grd, blk, 0, stream_>>>(
            d_A_.get(), d_b_.get(), d_g_.get(), P_, damping);

        // 2) Copy grad → rhs (potrs overwrites RHS in-place).
        CUDA_CHECK(cudaMemcpyAsync(d_rhs_.get(), d_grad,
            (size_t)P_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

        // 3) Cholesky factorise:  g = L Lᵀ   (lower triangle).
        CUSOLVER_CHECK(cusolverDnSpotrf(
            cusolver_, CUBLAS_FILL_MODE_LOWER, P_,
            d_g_.get(), P_,
            d_work_.get(), lwork_, d_info_.get()));

        // 4) Solve L Lᵀ Δ = rhs   (1 RHS, in-place).
        CUSOLVER_CHECK(cusolverDnSpotrs(
            cusolver_, CUBLAS_FILL_MODE_LOWER, P_,
            1,
            d_g_.get(), P_,
            d_rhs_.get(), P_,
            d_info_.get()));

        // 5) Write Δ back into d_grad so the optimizer steps along it.
        CUDA_CHECK(cudaMemcpyAsync(d_grad, d_rhs_.get(),
            (size_t)P_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    }

private:
    int P_ = 0;
    long long N_ = 0;
    cudaStream_t stream_ = nullptr;
    bool ready_ = false;

    DeviceBuffer<float2> d_mu_;     // [P, N] complex parameter-derivative vectors
    DeviceBuffer<float2> d_A_;      // [P, P] complex
    DeviceBuffer<float2> d_b_;      // [P]    complex
    DeviceBuffer<float>  d_g_;      // [P, P] real FIM (SPD)
    DeviceBuffer<float>  d_rhs_;    // [P]    real, holds grad → Δ
    DeviceBuffer<int>    d_info_;
    DeviceBuffer<float>  d_work_;   // cuSOLVER potrf workspace
    int lwork_ = 0;

    cublasHandle_t      cublas_   = nullptr;
    cusolverDnHandle_t  cusolver_ = nullptr;
};
