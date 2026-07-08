#define QHPC_CUDA
#include "QHPC_qml_core_v19.hpp"
#include "QHPC_async_pipeline_v19.hpp"
#include "QHPC_reduction_v19.hpp"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda::wmma;

// ============================================================================
// Tensor-core fused multi-qubit block — v15.
// ----------------------------------------------------------------------------
// Upgrades vs v14:
//
//  1. cp.async double-buffer staging: while the consumer warp feeds the MMA
//     pipeline from stage[buf], the producer issues cp.async loads into
//     stage[buf^1]. Hides the global-mem latency on the next tile.
//
//  2. Proper 3-MMA complex GEMM. Both A_im and A_neg_im are uploaded once;
//     no per-launch sign flip:
//        C_re = A_re*B_re + A_neg_im*B_im
//        C_im = A_re*B_im + A_im    *B_re
//
//  3. TILES_PER_BLOCK groups per block to raise occupancy and reuse the gate
//     fragments across multiple batch slots.
// ============================================================================

namespace qhpc_tc {

constexpr int M=16, N=16, K=16;

struct TCGateConst {
    DeviceBuffer<__half> g_re;
    DeviceBuffer<__half> g_im;
    DeviceBuffer<__half> g_neg_im;
    TCGateConst():g_re(16*16),g_im(16*16),g_neg_im(16*16){}
    void upload(const float* re16, const float* im16, cudaStream_t s){
        std::vector<__half> hre(16*16,(__half)0.f),
                            him(16*16,(__half)0.f),
                            hni(16*16,(__half)0.f);
        for(int i=0;i<4;++i)for(int j=0;j<4;++j){
            int dst=i*16+j;
            hre[dst]=__float2half( re16[i*4+j]);
            him[dst]=__float2half( im16[i*4+j]);
            hni[dst]=__float2half(-im16[i*4+j]);
        }
        g_re.upload(hre.data(),(int)hre.size(),s);
        g_im.upload(him.data(),(int)him.size(),s);
        g_neg_im.upload(hni.data(),(int)hni.size(),s);
    }
};

constexpr int TILES_PER_BLOCK = 4;

__global__ __launch_bounds__(128, 4)
void tc_complex_4q_block_async_f16(
    float2* __restrict__ sv,
    const __half* __restrict__ g_re,
    const __half* __restrict__ g_im,
    const __half* __restrict__ g_neg_im,
    int q0, int q1, int q2, int q3,
    long long n_groups)
{
    const long long base_group = (long long)blockIdx.x * TILES_PER_BLOCK;
    if(base_group >= n_groups) return;

    __shared__ alignas(16) __half stage_re[2][16*16];
    __shared__ alignas(16) __half stage_im[2][16*16];
    __shared__ float out_re_smem[16*16];
    __shared__ float out_im_smem[16*16];

    const int warp_id = threadIdx.x >> 5;

    // Zero-pad rows 1..15 of both stages once at kernel entry.
    for(int idx = threadIdx.x; idx < 16*16; idx += blockDim.x){
        int r = idx / 16;
        if(r > 0){
            stage_re[0][idx] = __float2half(0.f);
            stage_im[0][idx] = __float2half(0.f);
            stage_re[1][idx] = __float2half(0.f);
            stage_im[1][idx] = __float2half(0.f);
        }
    }
    __syncthreads();

    // Gate fragments in registers, loaded once per block.
    fragment<matrix_a, M,N,K, __half, row_major> a_re, a_im, a_negim;
    fragment<matrix_b, M,N,K, __half, col_major> b_re, b_im;
    fragment<accumulator, M,N,K, float> c_re, c_im;
    if(warp_id == 0){
        load_matrix_sync(a_re,    g_re,     16);
        load_matrix_sync(a_im,    g_im,     16);
        load_matrix_sync(a_negim, g_neg_im, 16);
    }

    auto support_index = [&](long long gid, int b4) -> long long {
        long long b = gid;
        long long lo = b & ((1LL<<q0)-1), hi = b >> q0;
        long long acc = (hi<<(q0+1)) | lo;
        lo = acc & ((1LL<<q1)-1); hi = acc >> q1;
        acc = (hi<<(q1+1)) | lo;
        lo = acc & ((1LL<<q2)-1); hi = acc >> q2;
        acc = (hi<<(q2+1)) | lo;
        lo = acc & ((1LL<<q3)-1); hi = acc >> q3;
        long long base = (hi<<(q3+1)) | lo;
        return base
            | (((b4>>0)&1) ? (1LL<<q0) : 0)
            | (((b4>>1)&1) ? (1LL<<q1) : 0)
            | (((b4>>2)&1) ? (1LL<<q2) : 0)
            | (((b4>>3)&1) ? (1LL<<q3) : 0);
    };

    auto stage_psi_async = [&](long long gid, int buf){
        // First 16 lanes cooperatively cp.async one float2 each into row 0
        // of stage_re/stage_im. We pack real into stage_re[buf][0..15] and
        // imag into stage_im[buf][0..15].
        if(threadIdx.x < 16){
            long long idx = support_index(gid, threadIdx.x);
            const float2* src = &sv[idx];
            // Stage real part: copy 4 bytes (x) followed by zero pad implicit.
            // We use 8B cp.async to bring x+y together into a uint2 staging
            // slot, then split. The staging layout puts real in stage_re row 0
            // and imag in stage_im row 0 via a quick post-process below.
            cp_async_8B(&stage_re[buf][threadIdx.x * 1], src);
        }
    };

    int buf = 0;
    stage_psi_async(base_group, buf);
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    // Post-process: split the float2 staged in stage_re row 0 into proper
    // stage_re/stage_im halves. Each lane handles its own slot.
    if(threadIdx.x < 16){
        __half2* slot = reinterpret_cast<__half2*>(
            &stage_re[buf][threadIdx.x]);
        // stage_re holds the raw float2 in 8 bytes; reinterpret to extract.
        float2 v = *reinterpret_cast<float2*>(slot);
        stage_re[buf][threadIdx.x] = __float2half(v.x);
        stage_im[buf][threadIdx.x] = __float2half(v.y);
    }
    __syncthreads();

    for(int t=0; t<TILES_PER_BLOCK; ++t){
        long long gid = base_group + t;
        int nxt = buf ^ 1;
        if(t + 1 < TILES_PER_BLOCK && (base_group + t + 1) < n_groups){
            stage_psi_async(base_group + t + 1, nxt);
            cp_async_commit();
        }
        if(gid >= n_groups) break;

        if(warp_id == 0){
            fill_fragment(c_re, 0.f);
            fill_fragment(c_im, 0.f);
            load_matrix_sync(b_re, stage_re[buf], 16);
            load_matrix_sync(b_im, stage_im[buf], 16);
            mma_sync(c_re, a_re,    b_re, c_re);
            mma_sync(c_re, a_negim, b_im, c_re);
            mma_sync(c_im, a_re,    b_im, c_im);
            mma_sync(c_im, a_im,    b_re, c_im);
            store_matrix_sync(out_re_smem, c_re, 16, mem_row_major);
            store_matrix_sync(out_im_smem, c_im, 16, mem_row_major);
        }
        __syncthreads();

        if(threadIdx.x < 16){
            long long idx = support_index(gid, threadIdx.x);
            sv[idx] = make_float2(out_re_smem[0*16 + threadIdx.x],
                                  out_im_smem[0*16 + threadIdx.x]);
        }

        cp_async_wait_group<0>();
        __syncthreads();
        if(t + 1 < TILES_PER_BLOCK){
            // The split for the newly arrived tile happens here, then loop
            // continues. Each lane splits its own slot.
            if(threadIdx.x < 16){
                __half2* slot = reinterpret_cast<__half2*>(
                    &stage_re[nxt][threadIdx.x]);
                float2 v = *reinterpret_cast<float2*>(slot);
                stage_re[nxt][threadIdx.x] = __float2half(v.x);
                stage_im[nxt][threadIdx.x] = __float2half(v.y);
            }
            __syncthreads();
        }
        buf = nxt;
    }
}

inline void launch_tc_complex_4q(
    float2* sv, const TCGateConst& g,
    int q0, int q1, int q2, int q3,
    long long n_groups, cudaStream_t s)
{
    int blocks = (int)((n_groups + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK);
    tc_complex_4q_block_async_f16<<<blocks, 128, 0, s>>>(
        sv, g.g_re.get(), g.g_im.get(), g.g_neg_im.get(),
        q0, q1, q2, q3, n_groups);
}

} // namespace qhpc_tc
