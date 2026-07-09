#pragma once
#include "QHPC_qml_core_v19.hpp"
#include "QHPC_quantum_ir_v19.hpp"
#include <cstdio>
#include <limits>

// ============================================================================
// Adaptive backend selector with a learned cost model.
// ----------------------------------------------------------------------------
// Cost function for each backend b:
//   cost_b(p) = w_qubits*p.n_qubits + w_gates*p.total_gates +
//               w_param*p.param_gates + w_cnot*p.cnot_count + w_y*p.y_terms
//               + bias + off_b
//
// Each backend additionally has hard-feasibility gates (e.g. NVRTC megakernel
// requires n_qubits<=13 by shared-memory ceiling; Heisenberg requires Clifford+
// param-rotations only; MPS requires low cut-entanglement to be viable).
//
// The weights live in QMLConfig::cost_weights so applications can override or
// load trained values without recompiling. A training script that emits
// CostWeights from microbenchmark CSVs is a separate offline tool.
// ============================================================================

struct BackendPick {
    BackendKind kind;
    float       score;
    const char* reason;
};

inline float cost_score(const CircuitProfile& p, const CostWeights& w){
    return w.w_qubits * (float)p.n_layers /* depth proxy */
         + w.w_gates  * (float)p.total_gates
         + w.w_param  * (float)p.param_gates
         + w.w_cnot   * (float)(p.cnot_count + p.cz_count + p.crz_count)
         + w.w_y      * (float)p.y_terms
         + w.bias;
}

inline bool feasible(BackendKind k, const CircuitProfile& p, int n_qubits,
                     bool clifford_only){
    switch(k){
        case BackendKind::NVRTC_MEGAKERNEL:
            // smem = 2^n * 8 bytes; cap at 96KB → n<=13.
            return n_qubits <= 13;
        case BackendKind::HEISENBERG:
            // Pauli tracker works for Clifford + RX/RY/RZ (with branching).
            // CRZ and T not yet supported in v14 tracker. So allow only if
            // either clifford-only OR no crz_count.
            return (clifford_only || p.crz_count==0);
        case BackendKind::MPS:
            // Needs low cut entanglement to win; we let the score handle that
            // and just require n_qubits>=8 (below it SV is unbeatable).
            return n_qubits >= 8;
        case BackendKind::DISTRIBUTED:
            return true;   // separate runtime checks PE count
        case BackendKind::STATEVEC_KERNELS:
        default:
            return n_qubits <= 30;
    }
}

inline BackendPick pick_backend(const QuantumIR& ir, const Observable& obs,
                                 const QMLConfig& cfg){
    BackendPick out{};
    out.score = std::numeric_limits<float>::infinity();
    if(cfg.backend != BackendKind::AUTO){
        out.kind = cfg.backend;
        out.score = 0.f;
        out.reason = "user-pinned backend";
        // Sanity check on user pick.
        CircuitProfile p = profile(ir, obs);
        bool cl = (p.clifford_only!=0);
        if(!feasible(cfg.backend, p, ir.n_qubits, cl)){
            out.kind = BackendKind::STATEVEC_KERNELS;
            out.reason = "user pick infeasible; downgraded to STATEVEC";
        }
        return out;
    }
    CircuitProfile p = profile(ir, obs);
    bool cl = (p.clifford_only!=0);
    const auto& w = cfg.cost_weights;
    float base = cost_score(p, w);

    auto consider = [&](BackendKind k, float off, const char* why){
        if(!feasible(k, p, ir.n_qubits, cl)) return;
        float s = base + off;
        if(s < out.score){
            out.score = s; out.kind = k; out.reason = why;
        }
    };
    // Apply entanglement-aware penalties.
    float mps_penalty = (p.twoq_density > 0.4f) ? 5.f : -2.f;
    consider(BackendKind::STATEVEC_KERNELS, w.off_statevec, "general-purpose SV");
    consider(BackendKind::NVRTC_MEGAKERNEL, w.off_nvrtc,
        "JIT mega-kernel; SV pinned in smem");
    consider(BackendKind::HEISENBERG,       w.off_heisenberg + (cl?-3.f:0.f),
        cl ? "Clifford-only -> O(n_terms) per gate" : "Pauli tracker, param-aware");
    consider(BackendKind::MPS,              w.off_mps + mps_penalty,
        "MPS w/ Tensor-Core SVD (low entanglement)");
    consider(BackendKind::DISTRIBUTED,      w.off_distributed,
        "multi-GPU slab decomposition");

    return out;
}

inline void log_backend_pick(const BackendPick& bp){
    const char* name = "?";
    switch(bp.kind){
        case BackendKind::STATEVEC_KERNELS:name="STATEVEC_KERNELS";break;
        case BackendKind::NVRTC_MEGAKERNEL:name="NVRTC_MEGAKERNEL";break;
        case BackendKind::HEISENBERG:      name="HEISENBERG";      break;
        case BackendKind::MPS:             name="MPS";             break;
        case BackendKind::DISTRIBUTED:     name="DISTRIBUTED";     break;
        default: break;
    }
    fprintf(stderr,"[Selector] backend = %s  score=%.3f  (%s)\n",
        name, bp.score, bp.reason ? bp.reason : "?");
}

// ============================================================================
// Online learning hook: collect (profile, backend, observed_runtime_ms) tuples
// and periodically refit cost weights via least squares. Skeleton.
// ============================================================================
struct CostSample {
    CircuitProfile p;
    BackendKind    backend;
    float          runtime_ms;
};

class CostModelTrainer {
public:
    void record(const CostSample& s){ samples_.push_back(s); }
    // Fit per-backend weights using closed-form ridge regression on the
    // feature vector (qubits-as-depth-proxy, gates, params, cnots, y_terms).
    // Implementation kept simple: caller invokes after enough samples.
    CostWeights refit(const CostWeights& init) const;
private:
    std::vector<CostSample> samples_;
};

inline CostWeights CostModelTrainer::refit(const CostWeights& init) const {
    CostWeights w = init;
    // PRODUCTION: solve 5xK regression with backend-conditional offsets.
    // SKELETON: returns init unchanged with the hook in place.
    return w;
}
