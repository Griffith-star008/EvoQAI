#pragma once
// ============================================================================
// QHPC v16 vectorized state-vector kernels (float4 fast-path)
// ----------------------------------------------------------------------------
// Memory-side optimization. On Ampere/Ada the L2 cache sector is 32 B; the
// PTX LD.E.128 / ST.E.128 instructions transfer 16 B in a single SASS op.
//
// For a 1q gate on qubit `q`, each thread processes one amplitude pair
// (sv[i0], sv[i1]) where i1 = i0 ^ (1<<q).
//
//   * q == 0 (LSB qubit): i0 = 2k, i1 = 2k+1. The two complex amps are
//     CONTIGUOUS — exactly 16 bytes. We reinterpret_cast<float4*> and issue
//     a single 128-bit load + a single 128-bit store. Throughput nearly
//     doubles on bandwidth-bound kernels.
//
//   * q >= 1: i0 and i1 are stride-(1<<q) apart -> NOT contiguous, so we
//     can't load the PAIR as float4. Instead we let each thread handle TWO
//     pairs (t and t+blockDim) and load two float2 from i0/i0+stride as
//     a single float4 = (a0.x, a0.y, a1.x, a1.y) where a0=sv[i0_t] and
//     a1=sv[i0_{t+1}]. That's also LD.E.128 throughput, applied to the
//     "same side" of the pair. Both i0-side and i1-side are loaded in
//     this fashion. Net effect: still 128-bit memory transactions.
//
// Param-baking-safe: all kernels read theta from d_params[param_idx] at
// execution time, same as v13+.
//
// Atomic-free reductions are unaffected (no reduction in these kernels).
// ============================================================================
#include "QHPC_qml_core_v19.hpp"

namespace qhpc_v16 {

constexpr int BSZ = 256;

// ----------------------------------------------------------------------------
// LSB fast path: q == 0. Pair is contiguous -> single float4 load/store.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void rotate_pair_rx(
    float2& p0, float2& p1, float c, float s)
{
    float2 n0 = make_float2( c*p0.x + s*p1.y,  c*p0.y - s*p1.x);
    float2 n1 = make_float2( s*p0.y + c*p1.x, -s*p0.x + c*p1.y);
    p0 = n0; p1 = n1;
}
__device__ __forceinline__ void rotate_pair_ry(
    float2& p0, float2& p1, float c, float s)
{
    float2 n0 = make_float2( c*p0.x - s*p1.x,  c*p0.y - s*p1.y);
    float2 n1 = make_float2( s*p0.x + c*p1.x,  s*p0.y + c*p1.y);
    p0 = n0; p1 = n1;
}
__device__ __forceinline__ void rotate_pair_rz(
    float2& p0, float2& p1, float c, float s)
{
    float2 n0 = make_float2( c*p0.x + s*p0.y,  c*p0.y - s*p0.x);
    float2 n1 = make_float2( c*p1.x - s*p1.y,  c*p1.y + s*p1.x);
    p0 = n0; p1 = n1;
}

#define DEFINE_LSB_KERNEL(NAME, BODY) \
__global__ __launch_bounds__(BSZ,8) \
void NAME(float2* __restrict__ sv, \
          const float* __restrict__ d_params, int param_idx, \
          long long np) \
{ \
    long long t = (long long)blockIdx.x * BSZ + threadIdx.x; \
    if(t >= np) return; \
    float theta = d_params[param_idx]; \
    float c, s; __sincosf(theta * 0.5f, &s, &c); \
    float4* sv4 = reinterpret_cast<float4*>(sv); \
    float4 v = sv4[t];                       /* LD.E.128 */ \
    float2 p0 = make_float2(v.x, v.y); \
    float2 p1 = make_float2(v.z, v.w); \
    BODY; \
    float4 out = make_float4(p0.x, p0.y, p1.x, p1.y); \
    sv4[t] = out;                            /* ST.E.128 */ \
}

DEFINE_LSB_KERNEL(sv_apply_rx_q0_v16,  rotate_pair_rx(p0, p1, c,  s))
DEFINE_LSB_KERNEL(sv_apply_ry_q0_v16,  rotate_pair_ry(p0, p1, c,  s))
DEFINE_LSB_KERNEL(sv_apply_rz_q0_v16,  rotate_pair_rz(p0, p1, c,  s))
DEFINE_LSB_KERNEL(sv_apply_rx_q0_inv_v16, rotate_pair_rx(p0, p1, c, -s))
DEFINE_LSB_KERNEL(sv_apply_ry_q0_inv_v16, rotate_pair_ry(p0, p1, c, -s))
DEFINE_LSB_KERNEL(sv_apply_rz_q0_inv_v16, rotate_pair_rz(p0, p1, c, -s))

#undef DEFINE_LSB_KERNEL

// ----------------------------------------------------------------------------
// q >= 1: coalesced two-pairs-per-thread variant. Each thread handles pairs
// t and t+1, loading two adjacent float2 from i0-side as a single float4,
// and the matching pair from i1-side similarly. This issues LD.E.128 even
// though the (i0, i1) within a single pair is not contiguous.
// ----------------------------------------------------------------------------
template<int GATE_ID, int INV>   // GATE_ID: 0=RX,1=RY,2=RZ ; INV: 0=fwd,1=adj
__device__ __forceinline__ void rotate_dispatch(
    float2& p0, float2& p1, float c, float s)
{
    if(INV) s = -s;
    if(GATE_ID == 0) rotate_pair_rx(p0, p1, c, s);
    if(GATE_ID == 1) rotate_pair_ry(p0, p1, c, s);
    if(GATE_ID == 2) rotate_pair_rz(p0, p1, c, s);
}

template<int GATE_ID, int INV>
__global__ __launch_bounds__(BSZ,8)
void sv_apply_param_general_v16(
    float2* __restrict__ sv,
    const float* __restrict__ d_params, int param_idx,
    int q, long long np)
{
    long long t2 = ((long long)blockIdx.x * BSZ + threadIdx.x) * 2;
    if(t2 >= np) return;
    float theta = d_params[param_idx];
    float c, s; __sincosf(theta * 0.5f, &s, &c);

    long long stride = 1LL << q;
    long long mask   = stride - 1;
    // Two pairs t2, t2+1. Their i0 indices are CONTIGUOUS in memory
    // iff (t2 & mask) != mask (i.e. they don't cross a stride boundary).
    // If they DO cross, we fall back to two scalar pair handles.
    bool can_vec = ((t2 & mask) != mask) && (t2 + 1 < np);

    if(can_vec){
        long long lo  = t2  & mask;
        long long hi  = t2  >> q;
        long long i0a = (hi << (q+1)) | lo;
        long long i1a = i0a | stride;
        long long i0b = i0a + 1;   // contiguous because (lo+1) hasn't wrapped
        long long i1b = i1a + 1;
        // i0a and i0b are adjacent in memory: 2 * float2 = 1 * float4.
        float4* sv4 = reinterpret_cast<float4*>(sv);
        float4 v0  = sv4[i0a >> 1];         // covers (sv[i0a], sv[i0a+1])
        float4 v1  = sv4[i1a >> 1];         // covers (sv[i1a], sv[i1a+1])
        float2 p0a = make_float2(v0.x, v0.y);
        float2 p0b = make_float2(v0.z, v0.w);
        float2 p1a = make_float2(v1.x, v1.y);
        float2 p1b = make_float2(v1.z, v1.w);
        rotate_dispatch<GATE_ID, INV>(p0a, p1a, c, s);
        rotate_dispatch<GATE_ID, INV>(p0b, p1b, c, s);
        sv4[i0a >> 1] = make_float4(p0a.x, p0a.y, p0b.x, p0b.y);
        sv4[i1a >> 1] = make_float4(p1a.x, p1a.y, p1b.x, p1b.y);
    } else {
        // Boundary-straddling fallback: scalar pair handles.
        for(int k=0; k<2 && t2+k<np; ++k){
            long long t = t2 + k;
            long long lo = t & mask, hi = t >> q;
            long long i0 = (hi << (q+1)) | lo, i1 = i0 | stride;
            float2 p0 = sv[i0], p1 = sv[i1];
            rotate_dispatch<GATE_ID, INV>(p0, p1, c, s);
            sv[i0] = p0; sv[i1] = p1;
        }
    }
}

// Concrete instantiations.
__global__ void sv_apply_rx_v16(float2* sv, const float* dp, int pi, int q, long long np){
    long long t2 = ((long long)blockIdx.x * BSZ + threadIdx.x);
    (void)t2;  // dispatcher: caller invokes the template directly.
}

// ----------------------------------------------------------------------------
// Host launchers — pick the LSB fast path when q==0, otherwise template.
// ----------------------------------------------------------------------------
inline void launch_sv_apply_rx_v16(float2* sv, const float* d_params, int param_idx,
                                    int q, long long np, cudaStream_t s, bool inverse){
    if(q == 0){
        int grid = (int)((np + BSZ - 1) / BSZ);
        if(inverse) sv_apply_rx_q0_inv_v16<<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
        else        sv_apply_rx_q0_v16    <<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
    } else {
        long long n2 = (np + 1) / 2;
        int grid = (int)((n2 + BSZ - 1) / BSZ);
        if(inverse) sv_apply_param_general_v16<0,1><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
        else        sv_apply_param_general_v16<0,0><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
    }
}
inline void launch_sv_apply_ry_v16(float2* sv, const float* d_params, int param_idx,
                                    int q, long long np, cudaStream_t s, bool inverse){
    if(q == 0){
        int grid = (int)((np + BSZ - 1) / BSZ);
        if(inverse) sv_apply_ry_q0_inv_v16<<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
        else        sv_apply_ry_q0_v16    <<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
    } else {
        long long n2 = (np + 1) / 2;
        int grid = (int)((n2 + BSZ - 1) / BSZ);
        if(inverse) sv_apply_param_general_v16<1,1><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
        else        sv_apply_param_general_v16<1,0><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
    }
}
inline void launch_sv_apply_rz_v16(float2* sv, const float* d_params, int param_idx,
                                    int q, long long np, cudaStream_t s, bool inverse){
    if(q == 0){
        int grid = (int)((np + BSZ - 1) / BSZ);
        if(inverse) sv_apply_rz_q0_inv_v16<<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
        else        sv_apply_rz_q0_v16    <<<grid,BSZ,0,s>>>(sv, d_params, param_idx, np);
    } else {
        long long n2 = (np + 1) / 2;
        int grid = (int)((n2 + BSZ - 1) / BSZ);
        if(inverse) sv_apply_param_general_v16<2,1><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
        else        sv_apply_param_general_v16<2,0><<<grid,BSZ,0,s>>>(sv, d_params, param_idx, q, np);
    }
}

} // namespace qhpc_v16
