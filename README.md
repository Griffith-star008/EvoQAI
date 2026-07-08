# Self-Evolving Quantum Machine Learning Framework for Adaptive AIOT

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![CUDA Version](https://img.shields.io/badge/CUDA-12.0-orange)](#)

This repository contains the foundational source code for the PhD research: **Self-Evolving Quantum AIOT Framework**. It bridges high-performance computing (via `QuaHPC`) with an autonomous meta-learning intelligence layer capable of mutating its own quantum circuits to adapt to concept drift in Edge AI environments.

---

## 🔬 Research Directions Implemented

| Module | Description | Location |
|---|---|---|
| **Direction 1** | Dynamic Quantum Representation Learning | `intelligence/representation/` |
| **Direction 2** | Evolutionary Quantum Intermediate Rep (IR) | `core/quantum_ir/` |
| **Direction 3** | Context-Aware Quantum Backend Selector | `core/backend_selector/` |
| **Direction 4 & 7** | Autonomous Evolution Engine & Runtime | `intelligence/evolution_engine/` |
| **Direction 5** | Semantic AIOT Intelligence | `intelligence/context_engine/` |
| **Direction 6** | Adaptive Quantum Feature Space | `intelligence/feature_space/` |
| **Direction 8** | Hybrid Classical-Quantum Intelligence | `hybrid/` |
| **Direction 9** | Lifelong Quantum AIOT Memory | `memory/` |
| **Direction 10** | Multi-Agent Self-Evolving Ecosystem | `agents/` |

---

## 🏛️ Architecture Overview

The system is built on a **C++ / Python Hybrid Architecture**:

1. **C++/CUDA Core (`QuaHPC_Core`)**: Handles ultra-fast quantum statevector simulations, tensor networks, and adjoint differentiation. 
2. **Python Intelligence Layer**: Acts as the "Operating System" that monitors accuracy, translates raw data into semantic contexts, and triggers the Evolution Engine.
3. **FFI Bridge**: Connects the Python intelligence layer to the C++ core via PyBind11, ensuring zero overhead during quantum execution.

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- CUDA Toolkit 12.0+ (For hardware acceleration)
- `numpy`, `scipy`

### 2. Run the End-to-End Simulation
To see the entire framework in action (from data representation to concept drift detection and circuit mutation), run the root experiment script:

```bash
python run_experiments.py
```

### 3. Run the Multi-Agent Orchestrator
To observe the decoupled background agents working together:

```bash
python agents/agent_orchestrator.py
```

---

*Author: Huy Ngo Anh (Independent Researcher)*  
*Contact: huyngoanh3@gmail.com*
