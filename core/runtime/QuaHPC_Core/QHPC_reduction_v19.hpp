#pragma once
#include <cuda_runtime.h>
#include "QHPC_memory_v10.hpp"

// ============================================================================
// QHPC hierarchical, atomic-free gradient reduction
// ----------------------------------------------------------------------------
// Adjoint kernels in v14 used atomicAdd(grad + param_idx, local) at block
// granularity. When many blocks land on the SAME param_idx (e.g. a 2-qubit
// gate whose grad-contribution is summed across the entire state-vector
// pair-space), the atomic becomes a serialization point.
//
// The v15 flow eliminates the per-block atomic by writing into a per-block
// partials buffer of shape (n_blocks, n_params) and reducing along the
// blocks axis in a separate cheap kernel. This drops contention to zero
// and lets the WMMA / fused kernels run at full memory bandwidth.
//
// Cost: one extra global write per block per param-touched + one O(blocks)
// reduction kernel. For typical training configurations the trade is a
// clear net win (atomic contention dominated).
//
// Memory budget: blocks * n_params * 4 bytes. For n_params=200 and
// blocks=2048 that's 1.6 MB — trivial.
//
// API:
//   - Allocate one ReductionScratch per (engine, max_blocks, n_params).
//   - Adjoint kernel writes a single float per (block, param_idx) into
//     scratch.partials[block * n_params + param_idx].
//   - Caller invokes reduce_partials_kernel to fold partials -> d_grads.
// ============================================================================

struct ReductionScratch {
    int n_params=0;
    int max_blocks=0;
    DeviceBuffer<float> partials;  // [max_blocks * n_params]
    void resize(int blocks, int params){
        max_blocks = blocks;
        n_params   = params;
        partials.resize((size_t)blocks * params);
    }
    void zero(cudaStream_t s){ partials.zero(s); }
    size_t bytes() const { return (size_t)max_blocks * n_params * sizeof(float); }
};

// ----------------------------------------------------------------------------
// Block-sum kernel. Each output element folds n_blocks_eff contributions
// belonging to one parameter. We launch one block per param; threads inside
// the block walk the partials column.
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void reduce_partials_to_grad_kernel(
    const float* __restrict__ partials,   // [n_blocks_eff, n_params]
    float*       __restrict__ d_grad,     // [n_params]
    int n_blocks_eff, int n_params)
{
    int p = blockIdx.x;
    if(p >= n_params) return;
    float acc = 0.f;
    for(int b = threadIdx.x; b < n_blocks_eff; b += blockDim.x){
        acc += partials[(size_t)b * n_params + p];
    }
    // Block reduction via shared mem.
    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for(int s = 128; s > 0; s >>= 1){
        if(threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if(threadIdx.x == 0) d_grad[p] = sm[0];
}

inline void reduce_partials_to_grad(
    ReductionScratch& s, float* d_grad, int n_blocks_eff,
    cudaStream_t stream)
{
    if(n_blocks_eff <= 0 || s.n_params <= 0) return;
    reduce_partials_to_grad_kernel<<<s.n_params, 256, 0, stream>>>(
        s.partials.get(), d_grad, n_blocks_eff, s.n_params);
}

// ----------------------------------------------------------------------------
// In-kernel block-local reduce helpers. These are atomic-free for the
// COMMON path (each block writes its own slot in partials); the only cross-
// block sync is the launch boundary of the reducer kernel.
// ----------------------------------------------------------------------------
__device__ __forceinline__ float qhpc_warp_sum(float v){
    #pragma unroll
    for(int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

__device__ __forceinline__ float qhpc_block_sum(float v){
    __shared__ float sm[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = qhpc_warp_sum(v);
    if(lane == 0) sm[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? sm[lane] : 0.f;
    if(wid == 0) v = qhpc_warp_sum(v);
    return v;
}

// Accumulate block's contribution into its dedicated slot. Multiple gates
// may share the same param_idx (parameter sharing across layers). Since
// each block writes only its own slot (no cross-block contention), one
// thread doing += is safe; no atomics needed.
__device__ __forceinline__ void qhpc_store_block_partial(
    float* partials, int n_params, int param_idx, float block_sum)
{
    if(threadIdx.x == 0){
        partials[(size_t)blockIdx.x * n_params + param_idx] += block_sum;
    }
}
