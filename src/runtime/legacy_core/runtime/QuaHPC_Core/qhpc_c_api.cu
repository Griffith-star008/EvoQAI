// ============================================================================
// QHPC C API implementation.
// Compiled into libqhpc.so along with the rest of v16.
// ============================================================================
#include "qhpc_c_api.h"
#include "QHPC_trainer_v19.cu"     // brings in the whole v16 stack
#include <cstring>
#include <memory>

struct QHPC_Circuit_ { QMLCircuit c; };
struct QHPC_Obs_     { Observable o; };
struct QHPC_Trainer_ { std::unique_ptr<QMLTrainer> t; };
struct QHPC_Metrics_ { QMLMetrics m; };

// ---------- Circuit ----------
extern "C" QHPC_Circuit* qhpc_circuit_new(int n_qubits){
    auto* c = new QHPC_Circuit_();
    c->c.n_qubits = n_qubits;
    return c;
}
extern "C" void qhpc_circuit_free(QHPC_Circuit* c){ delete c; }
extern "C" void qhpc_circuit_rx(QHPC_Circuit* c, int q, int pi){ c->c.rx(q, pi); }
extern "C" void qhpc_circuit_ry(QHPC_Circuit* c, int q, int pi){ c->c.ry(q, pi); }
extern "C" void qhpc_circuit_rz(QHPC_Circuit* c, int q, int pi){ c->c.rz(q, pi); }
extern "C" void qhpc_circuit_h (QHPC_Circuit* c, int q){ c->c.h(q); }
extern "C" void qhpc_circuit_cnot(QHPC_Circuit* c, int ctrl, int tgt){ c->c.cnot(ctrl, tgt); }
extern "C" void qhpc_circuit_cz  (QHPC_Circuit* c, int ctrl, int tgt){ c->c.cz(ctrl, tgt); }
extern "C" int qhpc_circuit_n_params(const QHPC_Circuit* c){ return c->c.n_params; }
extern "C" int qhpc_circuit_n_qubits(const QHPC_Circuit* c){ return c->c.n_qubits; }

// ---------- Observable ----------
extern "C" QHPC_Obs* qhpc_obs_new(){ return new QHPC_Obs_(); }
extern "C" void qhpc_obs_free(QHPC_Obs* o){ delete o; }
extern "C" void qhpc_obs_add_term(QHPC_Obs* o,
        uint32_t xm, uint32_t ym, uint32_t zm, float coeff){
    o->o.terms.push_back({xm, ym, zm, coeff});
}
extern "C" int qhpc_obs_n_terms(const QHPC_Obs* o){ return (int)o->o.terms.size(); }

// ---------- Config ----------
extern "C" void qhpc_config_default(QHPC_Config* out){
    QMLConfig def;
    out->n_epochs = def.n_epochs;
    out->lr = def.lr;
    out->beta1 = def.beta1;
    out->beta2 = def.beta2;
    out->eps_adam = def.eps_adam;
    out->grad_clip_norm = def.grad_clip_norm;
    out->use_cuda_graph = def.use_cuda_graph ? 1 : 0;
    out->print_every = def.print_every;
    out->fused_kernels = def.fused_kernels ? 1 : 0;
    out->async_loss = def.async_loss ? 1 : 0;
    out->backend = (int)def.backend;
    out->optimizer = (int)def.optimizer;
    out->adamw_enabled = def.adamw.enabled ? 1 : 0;
    out->adamw_weight_decay = def.adamw.weight_decay;
    out->verbose_ir = def.verbose_ir ? 1 : 0;
    out->verbose_selector = def.verbose_selector ? 1 : 0;
    out->use_qng = def.use_qng ? 1 : 0;
    out->qng_damping = def.qng_damping;
    out->qng_every = def.qng_every;
    out->use_tpb = def.use_tpb ? 1 : 0;
}

static QMLConfig _build_config(const QHPC_Config* c){
    QMLConfig out;
    out.n_epochs = c->n_epochs;
    out.lr = c->lr;
    out.beta1 = c->beta1;
    out.beta2 = c->beta2;
    out.eps_adam = c->eps_adam;
    out.grad_clip_norm = c->grad_clip_norm;
    out.use_cuda_graph = (c->use_cuda_graph != 0);
    out.print_every = c->print_every;
    out.fused_kernels = (c->fused_kernels != 0);
    out.async_loss = (c->async_loss != 0);
    out.backend = (BackendKind)c->backend;
    out.optimizer = (OptimizerKind)c->optimizer;
    out.adamw.enabled = (c->adamw_enabled != 0);
    out.adamw.weight_decay = c->adamw_weight_decay;
    out.verbose_ir = (c->verbose_ir != 0);
    out.verbose_selector = (c->verbose_selector != 0);
    out.use_qng = (c->use_qng != 0);
    out.qng_damping = c->qng_damping;
    out.qng_every = c->qng_every;
    out.use_tpb = (c->use_tpb != 0);
    return out;
}

// ---------- Trainer ----------
extern "C" QHPC_Trainer* qhpc_trainer_new(const QHPC_Circuit* c,
                                           const QHPC_Obs* o,
                                           const QHPC_Config* cfg)
{
    auto* h = new QHPC_Trainer_();
    QMLConfig qc = _build_config(cfg);
    // Trainer takes circuit by value; copy.
    QMLCircuit circ = c->c;
    Observable obs = o->o;
    h->t = std::make_unique<QMLTrainer>(std::move(circ), std::move(obs), qc);
    return h;
}
extern "C" void qhpc_trainer_free(QHPC_Trainer* h){ delete h; }

extern "C" int qhpc_trainer_train(QHPC_Trainer* h, float* loss_out, int cap,
                                   float* wall_s, float* best){
    QMLMetrics m = h->t->train();
    int n = (int)std::min((size_t)cap, m.loss_history.size());
    if(loss_out && n > 0) std::memcpy(loss_out, m.loss_history.data(), n*sizeof(float));
    if(wall_s) *wall_s = m.wall_time_s;
    if(best)   *best   = m.best_loss;
    return n;
}

extern "C" float qhpc_trainer_evaluate(QHPC_Trainer* /*h*/){
    // Not exposed in v16 trainer public API directly; integrator can wire it.
    return 0.f;
}

extern "C" int qhpc_trainer_n_params(const QHPC_Trainer* h){
    return (int)h->t->get_params().size();
}
extern "C" void qhpc_trainer_get_params(QHPC_Trainer* h, float* out){
    auto v = h->t->get_params();
    std::memcpy(out, v.data(), v.size()*sizeof(float));
}
extern "C" void qhpc_trainer_set_params(QHPC_Trainer* h, const float* in){
    auto v = h->t->get_params();
    std::vector<float> nv(v.size());
    std::memcpy(nv.data(), in, v.size()*sizeof(float));
    h->t->set_params(nv);
}
extern "C" int qhpc_trainer_active_backend(const QHPC_Trainer* h){
    return (int)h->t->active_backend();
}

extern "C" const char* qhpc_version(){ return "QHPC v19.1"; }
extern "C" const char* qhpc_build_info(){
    return "QHPC v19.1: graph-safe basis-state init + QNG-SGD step (Adam "
           "bypass) + device-side param-shift materialise + chain1q + "
           "Cholesky-QNG + TPB clique expectation + float4-vec SV + "
           "atomic-free grads + cuSOLVER persistent workspace";
}
