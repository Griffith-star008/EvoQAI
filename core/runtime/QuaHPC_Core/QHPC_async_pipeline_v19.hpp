#pragma once
// ============================================================================
// QHPC async pipeline primitives
// ----------------------------------------------------------------------------
// Thin device-side wrappers around the Ampere+ cp.async machinery. These let
// kernels overlap global-mem loads with compute by issuing async copies into
// shared memory and only awaiting them right before the consumer.
//
// Two flavors are exposed:
//
//   1. Raw cp.async wrappers       — for fine-grained control (used in WMMA
//                                    kernels that hand-tile their shared-mem
//                                    staging area).
//   2. cuda::pipeline scaffolds    — high-level 2-stage / 3-stage pipelines
//                                    that batch many copies per stage.
//
// On Volta/Turing (cc < 80) the wrappers fall back to a synchronous load so
// the same kernel compiles and runs (slower) on older GPUs.
// ============================================================================
#include <cuda_runtime.h>
#include <cstdint>

#if __CUDA_ARCH__ >= 800
  #define QHPC_HAVE_CP_ASYNC 1
  #include <cuda/pipeline>
  #include <cooperative_groups.h>
  #include <cooperative_groups/memcpy_async.h>
  namespace cg = cooperative_groups;
#else
  #define QHPC_HAVE_CP_ASYNC 0
#endif

// Copy `bytes` from `src` (global) to `dst` (shared) asynchronously. The
// caller MUST issue cp_async_commit() + cp_async_wait_group<N>() before
// using `dst`. `bytes` must be 4, 8 or 16. 16 gives the best throughput.
__device__ __forceinline__
void cp_async_16B(void* smem_dst, const void* gmem_src) {
#if QHPC_HAVE_CP_ASYNC
    uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem_src));
#else
    // Synchronous fallback: copy 16 bytes word-by-word.
    auto* d = reinterpret_cast<uint4*>(smem_dst);
    auto* s = reinterpret_cast<const uint4*>(gmem_src);
    *d = *s;
#endif
}

__device__ __forceinline__
void cp_async_8B(void* smem_dst, const void* gmem_src) {
#if QHPC_HAVE_CP_ASYNC
    uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_int), "l"(gmem_src));
#else
    auto* d = reinterpret_cast<uint2*>(smem_dst);
    auto* s = reinterpret_cast<const uint2*>(gmem_src);
    *d = *s;
#endif
}

// Commit all preceding cp.async instructions into a single named group.
__device__ __forceinline__ void cp_async_commit() {
#if QHPC_HAVE_CP_ASYNC
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

// Wait until all but the latest N cp.async groups have completed.
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if QHPC_HAVE_CP_ASYNC
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if QHPC_HAVE_CP_ASYNC
    asm volatile("cp.async.wait_all;\n");
#endif
}

// 16-B aligned copy of `n_16B` chunks from gmem to smem, one warp lane per chunk.
// Used by the WMMA staging code paths to load A and B tiles in 128-bit bursts.
__device__ __forceinline__
void cp_async_warp_burst_16B(void* smem_dst, const void* gmem_src, int n_16B,
                              int lane){
    const char* s = (const char*)gmem_src;
    char*       d = (char*)smem_dst;
    for(int k=lane; k<n_16B; k+=32){
        cp_async_16B(d + (size_t)k*16, s + (size_t)k*16);
    }
}

// ----------------------------------------------------------------------------
// 2-stage double-buffer scaffold using cuda::pipeline. Producer issues async
// loads into stage[buf]; consumer waits on stage[buf^1] before computing.
//
//   Usage pattern inside a kernel:
//     __shared__ alignas(16) char stage[2][TILE_BYTES];
//     auto block = cg::this_thread_block();
//     auto pipe  = cuda::make_pipeline();
//     int buf = 0;
//     // prime
//     produce_into(stage[buf], src_base);
//     pipe.producer_commit();
//     for(int k=0;k<NK;++k){
//         int nxt = buf ^ 1;
//         if(k+1<NK){ produce_into(stage[nxt], src_base+(k+1)*STRIDE);
//                     pipe.producer_commit(); }
//         pipe.consumer_wait();
//         consume(stage[buf]);
//         pipe.consumer_release();
//         buf = nxt;
//     }
// ----------------------------------------------------------------------------
