// ============================================================================
// qhpc_pybind.cpp  —  Host-only C++ Python bindings for QHPC v16.
//
// Built with MSVC (not nvcc). Links against libqhpc.dll (the CUDA core).
// Opaque QHPC handles are wrapped as Python capsules with custom deleters,
// avoiding any reliance on pybind11 type-traits over CUDA-internal types.
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qhpc_c_api.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

// ---------- Capsule helpers ----------
template<typename Deleter>
static py::capsule make_capsule(void* p, const char* name, Deleter d){
    return py::capsule(p, name, [](void* raw){
        // Bridge through a small trampoline because pybind11 capsule deleter
        // signature is void(void*).
        Deleter::run(raw);
    });
}

struct DelCircuit { static void run(void* p){ qhpc_circuit_free((QHPC_Circuit*)p); } };
struct DelObs     { static void run(void* p){ qhpc_obs_free((QHPC_Obs*)p); } };
struct DelTrainer { static void run(void* p){ qhpc_trainer_free((QHPC_Trainer*)p); } };

static QHPC_Circuit* get_circ(const py::capsule& c){
    if(std::strcmp(c.name(), "qhpc.Circuit") != 0)
        throw std::runtime_error("expected qhpc.Circuit capsule");
    return (QHPC_Circuit*)(void*)c;
}
static QHPC_Obs* get_obs(const py::capsule& c){
    if(std::strcmp(c.name(), "qhpc.Observable") != 0)
        throw std::runtime_error("expected qhpc.Observable capsule");
    return (QHPC_Obs*)(void*)c;
}
static QHPC_Trainer* get_trn(const py::capsule& c){
    if(std::strcmp(c.name(), "qhpc.Trainer") != 0)
        throw std::runtime_error("expected qhpc.Trainer capsule");
    return (QHPC_Trainer*)(void*)c;
}

PYBIND11_MODULE(qhpc, m){
    m.doc() = "QHPC v16: high-performance quantum-machine-learning runtime.";

    m.attr("STATEVEC")    = 0;
    m.attr("NVRTC")       = 1;
    m.attr("HEISENBERG")  = 2;
    m.attr("MPS")         = 3;
    m.attr("DISTRIBUTED") = 4;
    m.attr("AUTO")        = 255;
    m.attr("__version__") = qhpc_version();
    m.attr("BUILD_INFO")  = qhpc_build_info();

    // ---------- Circuit ----------
    m.def("Circuit", [](int n){
        QHPC_Circuit* p = qhpc_circuit_new(n);
        return py::capsule(p, "qhpc.Circuit", &DelCircuit::run);
    }, py::arg("n"));

    m.def("circuit_rx",   [](py::capsule c, int q, int pi){ qhpc_circuit_rx(get_circ(c), q, pi); });
    m.def("circuit_ry",   [](py::capsule c, int q, int pi){ qhpc_circuit_ry(get_circ(c), q, pi); });
    m.def("circuit_rz",   [](py::capsule c, int q, int pi){ qhpc_circuit_rz(get_circ(c), q, pi); });
    m.def("circuit_h",    [](py::capsule c, int q){ qhpc_circuit_h(get_circ(c), q); });
    m.def("circuit_cnot", [](py::capsule c, int a, int b){ qhpc_circuit_cnot(get_circ(c), a, b); });
    m.def("circuit_cz",   [](py::capsule c, int a, int b){ qhpc_circuit_cz(get_circ(c), a, b); });
    m.def("circuit_n_params", [](py::capsule c){ return qhpc_circuit_n_params(get_circ(c)); });
    m.def("circuit_n_qubits", [](py::capsule c){ return qhpc_circuit_n_qubits(get_circ(c)); });

    // ---------- Observable ----------
    m.def("Observable", [](){
        QHPC_Obs* p = qhpc_obs_new();
        return py::capsule(p, "qhpc.Observable", &DelObs::run);
    });
    m.def("obs_add_term", [](py::capsule o, uint32_t x, uint32_t y, uint32_t z, float coeff){
        qhpc_obs_add_term(get_obs(o), x, y, z, coeff);
    }, py::arg("obs"), py::arg("x_mask"), py::arg("y_mask"), py::arg("z_mask"), py::arg("coeff"));
    m.def("obs_n_terms", [](py::capsule o){ return qhpc_obs_n_terms(get_obs(o)); });

    // ---------- Config ----------
    py::class_<QHPC_Config>(m, "Config")
        .def(py::init([](){
            QHPC_Config c; qhpc_config_default(&c); return c;
        }))
        .def_readwrite("n_epochs", &QHPC_Config::n_epochs)
        .def_readwrite("lr", &QHPC_Config::lr)
        .def_readwrite("beta1", &QHPC_Config::beta1)
        .def_readwrite("beta2", &QHPC_Config::beta2)
        .def_readwrite("eps_adam", &QHPC_Config::eps_adam)
        .def_readwrite("grad_clip_norm", &QHPC_Config::grad_clip_norm)
        .def_readwrite("use_cuda_graph", &QHPC_Config::use_cuda_graph)
        .def_readwrite("print_every", &QHPC_Config::print_every)
        .def_readwrite("fused_kernels", &QHPC_Config::fused_kernels)
        .def_readwrite("async_loss", &QHPC_Config::async_loss)
        .def_readwrite("backend", &QHPC_Config::backend)
        .def_readwrite("optimizer", &QHPC_Config::optimizer)
        .def_readwrite("adamw_enabled", &QHPC_Config::adamw_enabled)
        .def_readwrite("adamw_weight_decay", &QHPC_Config::adamw_weight_decay)
        .def_readwrite("verbose_ir", &QHPC_Config::verbose_ir)
        .def_readwrite("verbose_selector", &QHPC_Config::verbose_selector)
        .def_readwrite("use_qng", &QHPC_Config::use_qng)
        .def_readwrite("qng_damping", &QHPC_Config::qng_damping)
        .def_readwrite("qng_every", &QHPC_Config::qng_every)
        .def_readwrite("use_tpb", &QHPC_Config::use_tpb);

    // ---------- Trainer ----------
    m.def("Trainer", [](py::capsule circ, py::capsule obs, const QHPC_Config& cfg){
        QHPC_Trainer* p = qhpc_trainer_new(get_circ(circ), get_obs(obs), &cfg);
        return py::capsule(p, "qhpc.Trainer", &DelTrainer::run);
    }, py::arg("circuit"), py::arg("observable"), py::arg("config"));

    m.def("trainer_train", [](py::capsule h){
        QHPC_Trainer* t = get_trn(h);
        int cap = 1 << 20;
        std::vector<float> buf(cap, 0.f);
        float wall=0.f, best=0.f;
        int n = qhpc_trainer_train(t, buf.data(), cap, &wall, &best);
        buf.resize(n);
        py::array_t<float> arr(n);
        if(n > 0) std::memcpy(arr.mutable_data(), buf.data(), n*sizeof(float));
        py::dict res;
        res["loss"]    = arr;
        res["wall_s"]  = wall;
        res["best"]    = best;
        res["backend"] = qhpc_trainer_active_backend(t);
        return res;
    });

    m.def("trainer_n_params", [](py::capsule h){ return qhpc_trainer_n_params(get_trn(h)); });

    m.def("trainer_get_params", [](py::capsule h){
        QHPC_Trainer* t = get_trn(h);
        int n = qhpc_trainer_n_params(t);
        py::array_t<float> arr(n);
        qhpc_trainer_get_params(t, arr.mutable_data());
        return arr;
    });

    m.def("trainer_set_params", [](py::capsule h, py::array_t<float, py::array::c_style|py::array::forcecast> arr){
        QHPC_Trainer* t = get_trn(h);
        int n = qhpc_trainer_n_params(t);
        if((int)arr.size() != n) throw std::runtime_error("param size mismatch");
        qhpc_trainer_set_params(t, arr.data());
    });

    m.def("trainer_active_backend", [](py::capsule h){ return qhpc_trainer_active_backend(get_trn(h)); });
}
