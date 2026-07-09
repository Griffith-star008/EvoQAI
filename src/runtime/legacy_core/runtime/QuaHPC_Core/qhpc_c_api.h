#pragma once
// ============================================================================
// QHPC C API — exported by libqhpc.so
// ----------------------------------------------------------------------------
// Stable C-linkage facade over the C++ trainer. Used by:
//   - pybind11 module (qhpc_python.cu)
//   - any other-language FFI (Rust, Julia, ...)
// ============================================================================
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QHPC_Circuit_  QHPC_Circuit;
typedef struct QHPC_Obs_      QHPC_Obs;
typedef struct QHPC_Trainer_  QHPC_Trainer;
typedef struct QHPC_Metrics_  QHPC_Metrics;

// Circuit construction.
QHPC_Circuit* qhpc_circuit_new(int n_qubits);
void          qhpc_circuit_free(QHPC_Circuit*);
void          qhpc_circuit_rx(QHPC_Circuit*, int q, int param_idx);
void          qhpc_circuit_ry(QHPC_Circuit*, int q, int param_idx);
void          qhpc_circuit_rz(QHPC_Circuit*, int q, int param_idx);
void          qhpc_circuit_h (QHPC_Circuit*, int q);
void          qhpc_circuit_cnot(QHPC_Circuit*, int ctrl, int tgt);
void          qhpc_circuit_cz  (QHPC_Circuit*, int ctrl, int tgt);
int           qhpc_circuit_n_params(const QHPC_Circuit*);
int           qhpc_circuit_n_qubits(const QHPC_Circuit*);

// Observable: list of Pauli terms encoded as (x_mask, y_mask, z_mask, coeff).
QHPC_Obs* qhpc_obs_new();
void      qhpc_obs_free(QHPC_Obs*);
void      qhpc_obs_add_term(QHPC_Obs*,
              uint32_t x_mask, uint32_t y_mask, uint32_t z_mask, float coeff);
int       qhpc_obs_n_terms(const QHPC_Obs*);

// Trainer config (a thin subset of QMLConfig; everything else stays default).
typedef struct {
    int   n_epochs;
    float lr;
    float beta1;
    float beta2;
    float eps_adam;
    float grad_clip_norm;
    int   use_cuda_graph;    // 0/1
    int   print_every;
    int   fused_kernels;     // 0/1
    int   async_loss;        // 0/1
    int   backend;           // 0=STATEVEC,1=NVRTC,2=HEIS,3=MPS,4=DIST,255=AUTO
    int   optimizer;         // 0=Adam, 1=AdamW
    int   adamw_enabled;     // 0/1
    float adamw_weight_decay;
    int   verbose_ir;
    int   verbose_selector;
    // ---- v17 ----
    int   use_qng;           // 0/1
    float qng_damping;
    int   qng_every;
    int   use_tpb;           // 0/1
} QHPC_Config;

void          qhpc_config_default(QHPC_Config* out);
QHPC_Trainer* qhpc_trainer_new(const QHPC_Circuit*, const QHPC_Obs*, const QHPC_Config*);
void          qhpc_trainer_free(QHPC_Trainer*);

// Run training. Caller passes a buffer for the loss history (size >= n_epochs);
// returns the actual number of recorded epochs (<= n_epochs if converged early).
int   qhpc_trainer_train(QHPC_Trainer*, float* loss_out, int loss_out_capacity,
                         float* out_wall_seconds, float* out_best_loss);

// One-shot evaluate (forward only) at the current parameters; returns expectation.
float qhpc_trainer_evaluate(QHPC_Trainer*);

// Param vector accessors.
int  qhpc_trainer_n_params(const QHPC_Trainer*);
void qhpc_trainer_get_params(QHPC_Trainer*, float* out);
void qhpc_trainer_set_params(QHPC_Trainer*, const float* in);

// Backend introspection — useful for benchmark logs.
int  qhpc_trainer_active_backend(const QHPC_Trainer*);

// Library version / build info.
const char* qhpc_version();
const char* qhpc_build_info();

#ifdef __cplusplus
}  // extern "C"
#endif
