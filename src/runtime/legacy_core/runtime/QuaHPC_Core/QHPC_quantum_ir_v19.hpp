#pragma once
#include "QHPC_qml_core_v19.hpp"
#include <unordered_map>
#include <cstdio>
#include <set>

// ============================================================================
// Quantum SSA IR + Dependency DAG + Alias Analysis
// ============================================================================

enum class IRKind : uint8_t {
    RX=0, RY=1, RZ=2, H=3, S=4, T=5, CNOT=6, CZ=7, CRZ=8,
    U1Q  = 64,
    NOP  = 200
};

struct IRInst {
    IRKind   kind   = IRKind::NOP;
    int      q0     = -1;
    int      q1     = -1;
    int      param_idx = -1;
    float    angle  = 0.f;
    float    u_re[4] = {0,0,0,0};
    float    u_im[4] = {0,0,0,0};
    bool     dead   = false;
    // DAG edges (filled by build_dag()):
    std::vector<int> preds;   // ops this depends on
    std::vector<int> succs;   // ops that depend on this
    int layer = -1;           // topological layer (parallel-schedule rank)
};

// Fusion cluster: a maximal sequence of consecutive single-qubit gates on the
// same qubit that can be collapsed into one 2x2 unitary.
struct FusionCluster {
    std::vector<int> inst_ids;   // ordered list of IR indices
    int qubit = -1;
};

struct QuantumIR {
    int n_qubits = 0;
    int n_params = 0;
    std::vector<IRInst> insts;
    // After build_dag:
    int n_layers = 0;
    std::vector<std::vector<int>> layers;   // layer[i] = ops at depth i
    // After alias analysis: per-qubit live range (first/last touching op).
    std::vector<int> qubit_first;
    std::vector<int> qubit_last;
    std::vector<FusionCluster> clusters;

    static QuantumIR from_circuit(const QMLCircuit& c){
        QuantumIR ir;
        ir.n_qubits = c.n_qubits;
        ir.n_params = c.n_params;
        ir.insts.reserve(c.gates.size());
        for(const auto& g : c.gates){
            IRInst i;
            switch(g.type){
                case GateType::RX:i.kind=IRKind::RX;break;
                case GateType::RY:i.kind=IRKind::RY;break;
                case GateType::RZ:i.kind=IRKind::RZ;break;
                case GateType::H: i.kind=IRKind::H; break;
                case GateType::S: i.kind=IRKind::S; break;
                case GateType::T: i.kind=IRKind::T; break;
                case GateType::CNOT:i.kind=IRKind::CNOT;break;
                case GateType::CZ:  i.kind=IRKind::CZ;  break;
                case GateType::CRZ: i.kind=IRKind::CRZ; break;
                default: i.kind=IRKind::NOP; break;
            }
            i.q0=g.qubits[0]; i.q1=g.qubits[1];
            i.param_idx=g.param_idx; i.angle=g.angle;
            ir.insts.push_back(i);
        }
        return ir;
    }

    QMLCircuit to_circuit() const {
        QMLCircuit c; c.n_qubits=n_qubits; c.n_params=n_params;
        for(const auto& i : insts){
            if(i.dead || i.kind==IRKind::NOP) continue;
            QMLGate g{};
            g.qubits[0]=i.q0; g.qubits[1]=i.q1;
            g.param_idx=i.param_idx; g.angle=i.angle;
            switch(i.kind){
                case IRKind::RX:g.type=GateType::RX;break;
                case IRKind::RY:g.type=GateType::RY;break;
                case IRKind::RZ:g.type=GateType::RZ;break;
                case IRKind::H: g.type=GateType::H; break;
                case IRKind::S: g.type=GateType::S; break;
                case IRKind::T: g.type=GateType::T; break;
                case IRKind::CNOT:g.type=GateType::CNOT;break;
                case IRKind::CZ:  g.type=GateType::CZ;  break;
                case IRKind::CRZ: g.type=GateType::CRZ; break;
                case IRKind::U1Q: g.type=GateType::H; break;  // fallback
                default: continue;
            }
            c.gates.push_back(g);
        }
        return c;
    }
};

// ============================================================================
// DAG construction: an op depends on the most-recent prior op that touched any
// of its qubits. This is the standard "last-writer" SSA dependency graph.
// Independent ops in the same layer can run in parallel (e.g. on multiple
// streams in eager mode, or as separately scheduled WMMA tiles).
// ============================================================================
inline void build_dag(QuantumIR& ir){
    const int N=(int)ir.insts.size();
    std::vector<int> last_on(ir.n_qubits, -1);
    for(int i=0;i<N;++i){
        IRInst& a = ir.insts[i];
        a.preds.clear(); a.succs.clear(); a.layer = 0;
        if(a.dead || a.kind==IRKind::NOP) continue;
        std::set<int> dep;
        if(a.q0>=0 && last_on[a.q0]>=0) dep.insert(last_on[a.q0]);
        if(a.q1>=0 && last_on[a.q1]>=0) dep.insert(last_on[a.q1]);
        for(int p : dep){
            a.preds.push_back(p);
            ir.insts[p].succs.push_back(i);
            a.layer = std::max(a.layer, ir.insts[p].layer + 1);
        }
        if(a.q0>=0) last_on[a.q0] = i;
        if(a.q1>=0) last_on[a.q1] = i;
    }
    // Materialize layer lists.
    int max_layer=0;
    for(const auto& i : ir.insts) if(!i.dead) max_layer=std::max(max_layer,i.layer);
    ir.n_layers = max_layer+1;
    ir.layers.assign(ir.n_layers, {});
    for(int idx=0; idx<N; ++idx){
        if(ir.insts[idx].dead || ir.insts[idx].kind==IRKind::NOP) continue;
        ir.layers[ir.insts[idx].layer].push_back(idx);
    }
}

// ============================================================================
// Alias analysis: qubit live range. Used by memory planners (e.g. an MPS
// scheduler that wants to free orphan-qubit tensors mid-circuit, or a tiled SV
// engine that can keep "dead" qubits in cold storage).
// ============================================================================
inline void analyze_qubit_liveness(QuantumIR& ir){
    ir.qubit_first.assign(ir.n_qubits, -1);
    ir.qubit_last.assign(ir.n_qubits, -1);
    for(int i=0;i<(int)ir.insts.size();++i){
        const auto& a = ir.insts[i];
        if(a.dead || a.kind==IRKind::NOP) continue;
        for(int q : {a.q0, a.q1}){
            if(q<0) continue;
            if(ir.qubit_first[q]<0) ir.qubit_first[q] = i;
            ir.qubit_last[q] = i;
        }
    }
}

// ============================================================================
// Fusion clustering: scan the DAG and detect chains of 1q gates on the same
// qubit with no intervening 2q op. Each cluster will be collapsed by the
// frontend into a single U1Q + handed to a tensor-core kernel or NVRTC megakernel.
// ============================================================================
inline void build_fusion_clusters(QuantumIR& ir){
    ir.clusters.clear();
    const int N=(int)ir.insts.size();
    std::vector<bool> seen(N,false);
    for(int i=0;i<N;++i){
        if(seen[i] || ir.insts[i].dead) continue;
        const IRInst& a = ir.insts[i];
        if(a.q1>=0) continue;  // 2q starts no cluster
        FusionCluster c; c.qubit=a.q0; c.inst_ids.push_back(i); seen[i]=true;
        // Extend forward greedily through 1q gates on same qubit, stopping at
        // any op that touches this qubit AND is 2-qubit.
        for(int j=i+1;j<N;++j){
            if(ir.insts[j].dead) continue;
            const IRInst& b = ir.insts[j];
            const bool touches = (b.q0==a.q0)||(b.q1==a.q0);
            if(!touches) continue;
            if(b.q1>=0) break;  // 2q closes the cluster
            c.inst_ids.push_back(j); seen[j]=true;
        }
        if(c.inst_ids.size()>=2) ir.clusters.push_back(c);
    }
}

// ============================================================================
// Linear passes (carried from v13).
// ============================================================================
inline int pass_cancel_involutions(QuantumIR& ir){
    int killed = 0;
    const int N = (int)ir.insts.size();
    for(int i=0;i<N;++i){
        if(ir.insts[i].dead) continue;
        IRInst& a = ir.insts[i];
        if(a.kind!=IRKind::H && a.kind!=IRKind::CNOT && a.kind!=IRKind::CZ) continue;
        for(int j=i+1;j<N;++j){
            if(ir.insts[j].dead) continue;
            IRInst& b = ir.insts[j];
            const bool touches_q0 = (b.q0==a.q0)||(b.q1>=0 && b.q1==a.q0);
            const bool touches_q1 = (a.q1>=0) && ((b.q0==a.q1)||(b.q1>=0 && b.q1==a.q1));
            if(!touches_q0 && !touches_q1) continue;
            if(b.kind==a.kind && b.q0==a.q0 && b.q1==a.q1){
                a.dead=true; b.dead=true; killed+=2;
            }
            break;
        }
    }
    return killed;
}

inline int pass_merge_rotations(QuantumIR& ir){
    int merged=0;
    const int N=(int)ir.insts.size();
    for(int i=0;i<N;++i){
        if(ir.insts[i].dead) continue;
        IRInst& a=ir.insts[i];
        if(a.kind!=IRKind::RX && a.kind!=IRKind::RY && a.kind!=IRKind::RZ) continue;
        if(a.param_idx>=0) continue;
        for(int j=i+1;j<N;++j){
            if(ir.insts[j].dead) continue;
            IRInst& b=ir.insts[j];
            const bool blocks = (b.q0==a.q0)||(b.q1>=0 && b.q1==a.q0);
            if(!blocks) continue;
            if(b.kind==a.kind && b.q0==a.q0 && b.param_idx<0){
                a.angle += b.angle;
                b.dead = true;
                ++merged;
            }
            break;
        }
    }
    return merged;
}

inline int pass_commute_diag_past_cnot(QuantumIR& ir){
    int moved=0;
    const int N=(int)ir.insts.size();
    for(int i=0;i<N-1;++i){
        if(ir.insts[i].dead) continue;
        IRInst& a=ir.insts[i];
        bool a_diag = (a.kind==IRKind::RZ || a.kind==IRKind::S || a.kind==IRKind::T);
        if(!a_diag || a.param_idx>=0) continue;
        IRInst& b=ir.insts[i+1];
        if(b.dead) continue;
        if(b.kind==IRKind::CNOT && b.q0==a.q0){
            std::swap(ir.insts[i], ir.insts[i+1]);
            ++moved;
        }
    }
    return moved;
}

// ============================================================================
// Profile: feeds into the learned cost model and the selector.
// ============================================================================
struct CircuitProfile {
    int  total_gates=0;
    int  param_gates=0;
    int  cnot_count=0;
    int  cz_count=0;
    int  crz_count=0;
    int  clifford_only=1;
    int  max_cut_entanglers=0;
    int  diag_z_count=0;
    int  n_layers=0;
    float twoq_density=0.f;
    int  y_terms=0;
};

inline CircuitProfile profile(const QuantumIR& ir, const Observable& obs){
    CircuitProfile p;
    int mid = ir.n_qubits/2;
    for(const auto& i : ir.insts){
        if(i.dead||i.kind==IRKind::NOP) continue;
        ++p.total_gates;
        if(i.param_idx>=0) ++p.param_gates;
        switch(i.kind){
            case IRKind::CNOT: ++p.cnot_count;
                if((i.q0<mid)!=(i.q1<mid)) ++p.max_cut_entanglers; break;
            case IRKind::CZ:   ++p.cz_count;
                if((i.q0<mid)!=(i.q1<mid)) ++p.max_cut_entanglers; break;
            case IRKind::CRZ:  ++p.crz_count;
                if((i.q0<mid)!=(i.q1<mid)) ++p.max_cut_entanglers; break;
            case IRKind::RZ:case IRKind::S:case IRKind::T:
                ++p.diag_z_count; break;
            default: break;
        }
        if(i.kind!=IRKind::H && i.kind!=IRKind::S && i.kind!=IRKind::CNOT && i.kind!=IRKind::CZ){
            // anything else is non-Clifford (RX/RY param, RZ param, T, CRZ)
            p.clifford_only = 0;
        }
    }
    p.n_layers = ir.n_layers;
    int twoq = p.cnot_count + p.cz_count + p.crz_count;
    p.twoq_density = p.total_gates ? (float)twoq / (float)p.total_gates : 0.f;
    for(const auto& t : obs.terms) if(t.y_mask) ++p.y_terms;
    return p;
}

inline void run_all_passes(QuantumIR& ir, bool verbose=false){
    int k1 = pass_cancel_involutions(ir);
    int k2 = pass_merge_rotations(ir);
    int k3 = pass_commute_diag_past_cnot(ir);
    int k4 = pass_merge_rotations(ir);
    build_dag(ir);
    analyze_qubit_liveness(ir);
    build_fusion_clusters(ir);
    if(verbose){
        size_t live = std::count_if(ir.insts.begin(),ir.insts.end(),
            [](const IRInst& i){return !i.dead && i.kind!=IRKind::NOP;});
        fprintf(stderr,
            "[IR] cancel=%d merge1=%d commute=%d merge2=%d  live=%zu  "
            "layers=%d  clusters=%zu\n",
            k1,k2,k3,k4,live,ir.n_layers,ir.clusters.size());
    }
}
