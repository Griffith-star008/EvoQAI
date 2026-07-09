#pragma once
#include "QHPC_memory_v10.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

enum class GateType : uint8_t {
    RX=0,RY=1,RZ=2,H=3,S=4,T=5,CNOT=6,CZ=7,CRZ=8,BARRIER=255
};
enum class PrecisionMode : uint8_t { FP32=0,FP16=1,BF16=2,FP8=3 };

enum class BackendKind : uint8_t {
    STATEVEC_KERNELS = 0,
    NVRTC_MEGAKERNEL = 1,
    HEISENBERG       = 2,
    MPS              = 3,
    DISTRIBUTED      = 4,
    AUTO             = 255   // selector decides
};

enum class OptimizerKind : uint8_t {
    ADAM   = 0,
    ADAMW  = 1,
    LBFGS  = 2
};

struct Gate1qMatrix { float g[8]; };
inline Gate1qMatrix gate_rx(float t){float c,s;sincosf(t*.5f,&s,&c);return{c,0,0,-s,0,-s,c,0};}
inline Gate1qMatrix gate_ry(float t){float c,s;sincosf(t*.5f,&s,&c);return{c,0,-s,0,s,0,c,0};}
inline Gate1qMatrix gate_rz(float t){float c,s;sincosf(t*.5f,&s,&c);return{c,-s,0,0,0,0,c,s};}
inline Gate1qMatrix gate_h(){static constexpr float r=0.70710678118f;return{r,0,r,0,r,0,-r,0};}
inline Gate1qMatrix gate_s(){return{1,0,0,0,0,0,0,1};}
inline Gate1qMatrix gate_t(){static constexpr float r=0.70710678118f;return{1,0,0,0,0,0,r,r};}

struct GPUPauliTerm { uint32_t x_mask=0,y_mask=0,z_mask=0; float coeff=0.f; };

struct QMLGate {
    GateType type=GateType::BARRIER;
    int      qubits[2]={-1,-1};
    float    angle=0.f;
    int      param_idx=-1;
    bool is_param()const{return param_idx>=0;}
};

struct QMLCircuit {
    int n_qubits=0,n_params=0;
    std::vector<QMLGate> gates;
    void rx(int q,int p,float a=0){gates.push_back({GateType::RX,{q,-1},a,p});n_params=std::max(n_params,p+1);}
    void ry(int q,int p,float a=0){gates.push_back({GateType::RY,{q,-1},a,p});n_params=std::max(n_params,p+1);}
    void rz(int q,int p,float a=0){gates.push_back({GateType::RZ,{q,-1},a,p});n_params=std::max(n_params,p+1);}
    void h(int q){gates.push_back({GateType::H,{q,-1},0,-1});}
    void cnot(int c,int t){gates.push_back({GateType::CNOT,{c,t},0,-1});}
    void cz(int c,int t){gates.push_back({GateType::CZ,{c,t},0,-1});}
    void crz(int c,int t,int p,float a=0){gates.push_back({GateType::CRZ,{c,t},a,p});n_params=std::max(n_params,p+1);}
    static QMLCircuit hardware_efficient(int n,int layers){
        QMLCircuit c;c.n_qubits=n;int p=0;
        for(int l=0;l<layers;++l){
            for(int q=0;q<n;++q)c.ry(q,p++);
            for(int q=0;q<n;++q)c.rz(q,p++);
            for(int q=0;q<n-1;++q)c.cnot(q,q+1);
            if(n>2)c.cnot(n-1,0);
        }
        for(int q=0;q<n;++q)c.ry(q,p++);
        return c;
    }
    static QMLCircuit qaoa(int n,int p_layers,const std::vector<std::pair<int,int>>& E){
        QMLCircuit c;c.n_qubits=n;int p=0;
        for(int q=0;q<n;++q)c.h(q);
        for(int l=0;l<p_layers;++l){
            for(auto&[u,v]:E){c.cnot(u,v);c.rz(v,p++);c.cnot(u,v);}
            for(int q=0;q<n;++q)c.rx(q,p++);
        }
        return c;
    }
};

struct Observable {
    std::vector<GPUPauliTerm> terms;
    static Observable max_cut(int n,const std::vector<std::pair<int,int>>& E){
        Observable o;
        for(auto&[i,j]:E){o.terms.push_back({0,0,0,0.5f});o.terms.push_back({0,0,(1u<<i)|(1u<<j),-0.5f});}
        return o;
    }
    static Observable ising_zz(int n,const std::vector<std::pair<int,int>>& E,float J=1.f){
        Observable o;
        for(auto&[i,j]:E)o.terms.push_back({0,0,(1u<<i)|(1u<<j),J});
        return o;
    }
    static Observable pauli_z_sum(int n,float c=1.f){
        Observable o;
        for(int q=0;q<n;++q)o.terms.push_back({0,0,1u<<q,c});
        return o;
    }
};

struct NVRTCConfig {
    int  cc_major=8, cc_minor=0;
    bool keep_ptx=false;
    bool use_fast_math=true;
    bool half_for_obs=false;
};

struct MPSConfig {
    int  bond_dim=64;
    bool use_tensor_cores=true;
    bool fp16_truncate=true;
    // Bond dimensions are physically padded up to a multiple of this to satisfy
    // WMMA 16-byte alignment. 16 is the hard minimum for f16 fragments.
    int  alignment=16;
    // Adaptive truncation: drop singular values whose squared-sum below this
    // fraction of total. 0 disables adaptivity (fixed bond_dim).
    float truncation_eps=1e-6f;
};

struct DistConfig {
    int  n_local_qubits=-1;
    bool use_nvshmem=true;
    // Communication scheduler: when set, the partitioner clusters entangling
    // gates to minimize cross-PE chatter.
    bool gate_locality_aware=true;
};

struct AdamWConfig {
    bool  enabled=false;
    float weight_decay=0.01f;
};

struct GradScalerConfig {
    bool  enabled=false;
    float init_scale=65536.f;
    float growth_factor=2.f;
    float backoff_factor=0.5f;
    int   growth_interval=2000;
};

struct PersistentConfig {
    bool enabled=false;
    int  spin_backoff_ns=1000;   // host_seq poll backoff inside kernel
    int  watchdog_break_ms=1500; // safety: break loop after this if no signal
};

// Learned cost-model weights. Production: trained from microbenchmarks; here
// we initialize with hand-tuned defaults that match observed timings on Ada.
struct CostWeights {
    // y = w_qubits * n_qubits + w_gates * total_gates + w_param * param_gates
    //   + w_cnot * cnot_count + w_y * y_terms + bias
    float w_qubits=2.5f;
    float w_gates=0.1f;
    float w_param=0.3f;
    float w_cnot=0.4f;
    float w_y=0.5f;
    float bias=-3.f;
    // Per-backend offset (smaller = faster expected runtime).
    float off_statevec=0.f;
    float off_nvrtc=-2.f;       // mega-kernel is faster for small n
    float off_heisenberg=-1.f;  // wins when Clifford-heavy
    float off_mps=1.f;          // overhead unless low entanglement
    float off_distributed=3.f;  // worth it only when single-GPU OOM
};

struct QMLConfig {
    int   n_epochs        = 200;
    float lr              = 0.05f;
    float beta1           = 0.9f;
    float beta2           = 0.999f;
    float eps_adam        = 1e-8f;
    float grad_clip_norm  = 1.f;
    bool  use_cuda_graph  = true;
    int   print_every     = 10;
    float convergence_tol = 1e-6f;
    int   convergence_win = 20;
    PrecisionMode sv_prec = PrecisionMode::FP32;
    int   batch_size      = 1;
    bool  fused_kernels   = true;
    bool  async_loss      = true;

    BackendKind   backend    = BackendKind::AUTO;
    OptimizerKind optimizer  = OptimizerKind::ADAM;

    NVRTCConfig       nvrtc{};
    MPSConfig         mps{};
    DistConfig        dist{};
    AdamWConfig       adamw{};
    GradScalerConfig  grad_scaler{};
    PersistentConfig  persistent{};
    CostWeights       cost_weights{};

    // Logging hooks
    bool verbose_ir=true;
    bool verbose_selector=true;

    // ---- v17 Quantum Natural Gradient ----
    // When enabled, the trainer replaces ∇L with g⁻¹·∇L where g is the
    // Fubini-Study metric. Convergence in 5–10 epochs replaces 100+ epochs
    // for typical VQE / QML ansätze. Cost: 2·P forward passes per epoch.
    // ENABLE WHEN n_params is moderate (≲ 512) — solver is O(P³).
    bool  use_qng        = false;
    float qng_damping    = 1e-3f;   // Tikhonov ridge added to diag(g)
    int   qng_every      = 1;       // recompute metric every k epochs

    // ---- v17 TPB grouping ----
    bool  use_tpb        = false;
};

struct QMLMetrics {
    std::vector<float> loss_history,grad_norm_history;
    float best_loss=1e30f;
    int   best_epoch=-1;
    float wall_time_s=0.f,throughput_keps=0.f;
};

struct ScheduledOp {
    enum Kind : uint8_t {
        GATE_1Q_STATIC=0,
        GATE_1Q_PARAM =1,
        GATE_CNOT     =2,
        GATE_CZ       =3,
        GATE_CRZ      =4
    };
    Kind     kind;
    GateType type;
    int      q0=-1, q1=-1;
    int      param_idx=-1;
    int      gate_idx=-1;
};

struct GateSchedule {
    std::vector<ScheduledOp> fwd;
    std::vector<ScheduledOp> bwd;
    static GateSchedule build(const QMLCircuit& c){
        GateSchedule s;
        s.fwd.reserve(c.gates.size());
        for(int i=0;i<(int)c.gates.size();++i){
            const auto& g=c.gates[i];
            ScheduledOp op{};
            op.type=g.type; op.q0=g.qubits[0]; op.q1=g.qubits[1];
            op.param_idx=g.param_idx; op.gate_idx=i;
            switch(g.type){
                case GateType::CNOT: op.kind=ScheduledOp::GATE_CNOT; break;
                case GateType::CZ:   op.kind=ScheduledOp::GATE_CZ;   break;
                case GateType::CRZ:  op.kind=ScheduledOp::GATE_CRZ;  break;
                case GateType::RX:case GateType::RY:case GateType::RZ:
                    op.kind=ScheduledOp::GATE_1Q_PARAM; break;
                default:
                    op.kind=ScheduledOp::GATE_1Q_STATIC; break;
            }
            s.fwd.push_back(op);
        }
        s.bwd.assign(s.fwd.rbegin(), s.fwd.rend());
        return s;
    }
};

// ----------------------------------------------------------------------------
// v19: Chain detection.
// A "chain" is a maximal sequence of 1q-param gates acting on the SAME qubit
// without any other gate touching that qubit in between. Such a chain can
// execute as ONE kernel launch: load the amplitude pair once, run all
// rotations in registers, store once. Saves N-1 round trips and N-1 launches.
//
// Chains are encoded as ChainOp{q, len, axes[4], param_idx[4], signs[4]}.
// Length capped at 4 (matches the kernel template instantiations).
// ----------------------------------------------------------------------------
struct Chain1q {
    int  q          = -1;
    int  len        = 0;
    char axes[4]    = {0,0,0,0};      // 0=Rx, 1=Ry, 2=Rz
    int  param_idx[4]= {-1,-1,-1,-1};
    char signs[4]   = {1,1,1,1};      // +1 fwd, -1 adjoint
};

// One step in the chained schedule: EITHER a Chain1q (kind=CHAIN_1Q) OR a
// regular ScheduledOp (CNOT/CZ/CRZ/1q-static).
struct SchedStep {
    enum Kind { CHAIN_1Q=10, OP=11 } kind;
    Chain1q     chain;
    ScheduledOp op;
};

inline std::vector<SchedStep> build_chained_schedule(
    const std::vector<ScheduledOp>& ops, bool inverse_signs)
{
    std::vector<SchedStep> out;
    out.reserve(ops.size());

    auto axis_of = [](GateType t){
        if(t == GateType::RX) return 0;
        if(t == GateType::RY) return 1;
        return 2; // RZ
    };

    // Per-qubit "in-flight" chain. When we see a gate that touches the qubit
    // for any other reason, the chain flushes.
    int n_max_qubits = 1;
    for(const auto& o : ops){
        n_max_qubits = std::max({n_max_qubits, o.q0+1, o.q1+1});
    }
    std::vector<Chain1q> live(n_max_qubits);

    auto flush = [&](int q){
        if(live[q].len > 0){
            SchedStep s; s.kind = SchedStep::CHAIN_1Q; s.chain = live[q];
            out.push_back(s);
            live[q] = Chain1q{};
        }
    };

    for(const auto& op : ops){
        if(op.kind == ScheduledOp::GATE_1Q_PARAM){
            int q = op.q0;
            // Same-qubit, room left: extend chain.
            if(live[q].len == 0) live[q].q = q;
            if(live[q].q != q || live[q].len >= 4){
                flush(q); live[q].q = q;
            }
            int k = live[q].len;
            live[q].axes[k]      = (char)axis_of(op.type);
            live[q].param_idx[k] = op.param_idx;
            live[q].signs[k]     = inverse_signs ? -1 : 1;
            live[q].len = k + 1;
        } else {
            // Anything else touching q0 (and q1 for 2-qubit gates) flushes.
            flush(op.q0);
            if(op.q1 >= 0) flush(op.q1);
            SchedStep s; s.kind = SchedStep::OP; s.op = op;
            out.push_back(s);
        }
    }
    // Flush remaining live chains at end of schedule.
    for(int q = 0; q < (int)live.size(); ++q) flush(q);
    return out;
}

inline bool is_clifford_only(const QMLCircuit& c){
    for(const auto& g:c.gates){
        switch(g.type){
            case GateType::H:case GateType::S:case GateType::CNOT:case GateType::CZ:
                break;
            default: return false;
        }
    }
    return true;
}

inline int align_up(int x, int a){ return (x + a - 1) / a * a; }
