#pragma once
// ============================================================================
// QHPC_chain_1q_v19.cuh
// ----------------------------------------------------------------------------
// Chained 1q-rotation kernel.
//
// Pattern in hardware-efficient ansatze:
//     RY(q, θ1) → RZ(q, θ2) → (CNOT...) → RY(q, θ3) → RZ(q, θ4) → ...
//
// Two consecutive rotations on the SAME qubit with NO entangling gate in
// between can be applied together: one pair load, two rotations in
// registers, one pair store.  Cuts VRAM round-trips by 2× for those
// segments, and slashes per-gate kernel-launch overhead.
//
// Kernel handles chains up to length 4 (covers the common ansatz patterns
// RY+RZ, RX+RY+RZ).  Chain >4 splits across multiple launches.
//
// Encoding:
//   axes[k] = 0 (Rx), 1 (Ry), 2 (Rz);   sign[k] = +1 (forward) or -1 (adjoint)
//   theta[k] read from d_params[param_idx[k]] at execution time
//     (param-baking-safe; same as the single-gate kernels)
// ============================================================================
#include "QHPC_qml_core_v19.hpp"

namespace qhpc_v19 {

constexpr int CHAIN_BSZ = 256;

// Rotate a pair (p0, p1) by axis `ax` (0=Rx, 1=Ry, 2=Rz) at angle θ.
__device__ __forceinline__ void rot_pair_axis(
    float2& p0, float2& p1, int ax, float c, float s)
{
    if(ax == 0){
        float2 n0 = make_float2( c*p0.x + s*p1.y,  c*p0.y - s*p1.x);
        float2 n1 = make_float2( s*p0.y + c*p1.x, -s*p0.x + c*p1.y);
        p0 = n0; p1 = n1;
    } else if(ax == 1){
        float2 n0 = make_float2( c*p0.x - s*p1.x,  c*p0.y - s*p1.y);
        float2 n1 = make_float2( s*p0.x + c*p1.x,  s*p0.y + c*p1.y);
        p0 = n0; p1 = n1;
    } else {
        float2 n0 = make_float2( c*p0.x + s*p0.y,  c*p0.y - s*p0.x);
        float2 n1 = make_float2( c*p1.x - s*p1.y,  c*p1.y + s*p1.x);
        p0 = n0; p1 = n1;
    }
}

// Apply a chain of up to MAX_CHAIN rotations on qubit q, all in registers.
// q == 0 fast path: contiguous LD/ST.128 via float4 reinterpret.
template<int MAX_CHAIN>
__global__ __launch_bounds__(CHAIN_BSZ, 8)
void chain_apply_1q_q0_v19(
    float2* __restrict__ sv,
    const float* __restrict__ d_params,
    const int*   __restrict__ param_idx,    // [MAX_CHAIN]
    const char*  __restrict__ axes,         // [MAX_CHAIN]  0/1/2
    const char*  __restrict__ signs,        // [MAX_CHAIN]  +1/-1
    int chain_len,
    long long np)
{
    long long t = (long long)blockIdx.x * CHAIN_BSZ + threadIdx.x;
    if(t >= np) return;
    float4* sv4 = reinterpret_cast<float4*>(sv);
    float4 v = sv4[t];
    float2 p0 = make_float2(v.x, v.y);
    float2 p1 = make_float2(v.z, v.w);
    #pragma unroll
    for(int k = 0; k < MAX_CHAIN; ++k){
        if(k >= chain_len) break;
        float theta = d_params[param_idx[k]];
        float c, s; __sincosf(theta * 0.5f, &s, &c);
        if(signs[k] < 0) s = -s;
        rot_pair_axis(p0, p1, (int)axes[k], c, s);
    }
    sv4[t] = make_float4(p0.x, p0.y, p1.x, p1.y);
}

// q >= 1: pair stride 2^q apart; can't pack as float4 within a pair, so
// fall back to scalar load/store but STILL gain by applying all rotations
// in registers between load and store.
template<int MAX_CHAIN>
__global__ __launch_bounds__(CHAIN_BSZ, 8)
void chain_apply_1q_general_v19(
    float2* __restrict__ sv,
    const float* __restrict__ d_params,
    const int*   __restrict__ param_idx,
    const char*  __restrict__ axes,
    const char*  __restrict__ signs,
    int chain_len,
    int q,
    long long np)
{
    long long t = (long long)blockIdx.x * CHAIN_BSZ + threadIdx.x;
    if(t >= np) return;
    long long lo = t & ((1LL << q) - 1);
    long long hi = t >> q;
    long long i0 = (hi << (q+1)) | lo;
    long long i1 = i0 | (1LL << q);
    float2 p0 = sv[i0], p1 = sv[i1];
    #pragma unroll
    for(int k = 0; k < MAX_CHAIN; ++k){
        if(k >= chain_len) break;
        float theta = d_params[param_idx[k]];
        float c, s; __sincosf(theta * 0.5f, &s, &c);
        if(signs[k] < 0) s = -s;
        rot_pair_axis(p0, p1, (int)axes[k], c, s);
    }
    sv[i0] = p0; sv[i1] = p1;
}

// Host launcher: pick fast path by q, instantiate the right template.
inline void launch_chain_1q_v19(
    float2* sv, const float* d_params,
    const int* d_param_idx, const char* d_axes, const char* d_signs,
    int chain_len, int q, long long np, cudaStream_t stream)
{
    int grid = (int)((np + CHAIN_BSZ - 1) / CHAIN_BSZ);
    if(q == 0){
        switch(chain_len){
            case 1: chain_apply_1q_q0_v19<1><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 1, np); break;
            case 2: chain_apply_1q_q0_v19<2><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 2, np); break;
            case 3: chain_apply_1q_q0_v19<3><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 3, np); break;
            default: chain_apply_1q_q0_v19<4><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 4, np); break;
        }
    } else {
        switch(chain_len){
            case 1: chain_apply_1q_general_v19<1><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 1, q, np); break;
            case 2: chain_apply_1q_general_v19<2><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 2, q, np); break;
            case 3: chain_apply_1q_general_v19<3><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 3, q, np); break;
            default: chain_apply_1q_general_v19<4><<<grid,CHAIN_BSZ,0,stream>>>(
                sv, d_params, d_param_idx, d_axes, d_signs, 4, q, np); break;
        }
    }
}

} // namespace qhpc_v19
