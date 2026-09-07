# EvoQAI: Evolutionary Quantum Artificial Intelligence

EvoQAI is a research codebase designed to investigate **Evolutionary Quantum Circuit Adaptation under Concept Drift** and **Causal Digital Twin-guided Safe Evolution** for resource-constrained Edge/AIoT environments.

This repository focuses on overcoming the limitations of static Variational Quantum Circuits (VQCs) when deployed in non-stationary data streams, while strictly respecting the physical noise constraints of Noisy Intermediate-Scale Quantum (NISQ) hardware.

## Core Research Focus
This framework is currently driving two primary research investigations (targeting IEEE Tier-1 venues):
1. **Evolutionary Circuit Adaptation (Paper 1):** Dynamic appending/pruning of parameterized quantum layers based on real-time concept drift detection in streaming data (e.g., SEA streams).
2. **Causal Digital Twin Safety (Paper 2):** A Structural Causal Model (SCM) that predicts depth-induced hardware noise (depolarizing/relaxation) to preemptively block unsafe circuit mutations that would lead to fidelity collapse.

## Architecture
- **Quantum Core:** Pure Python implementation using **PennyLane** and **PyTorch**, supporting true adjoint differentiation, parameterized rotation gates, and statevector simulation.
- **Hardware Integration:** Support for real QPU execution via `qiskit-ibm-provider`, simulated local noise via `qiskit.aer`, and CPU-fallback via `default.qubit`.
- **Data Stream:** Custom SEA concept drift generator for non-stationary AIoT data.
- **Engine:** PyTorch-based online learning loop with sliding-window drift detection and a multi-objective context-aware Backend Selector.

## Getting Started

### Installation
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### Reproducing Experiments
To run the end-to-end experiment demonstrating the necessity of the Causal Digital Twin under concept drift:
```bash
bash scripts/reproduce_all.sh
# Or manually:
python experiments/run_paper1_drift.py
```
The output plots will be saved to `experiments/reports/`.

## Running on Real Hardware
Rename `credentials.env.template` to `credentials.env` and insert your IBM Quantum API Token. The backend selector can then autonomously route highly complex, drift-adapted circuits to IBM QPUs.

## Academic Reproducibility
This project enforces rigorous reproducibility. All baseline comparisons (Safe vs Unsafe evolution) use fixed seeds and deterministically generated drift points. 

*Disclaimer: This is an academic research prototype. Previous preliminary claims regarding C++/CUDA bindings and large-scale "production-readiness" have been explicitly removed to prioritize rigorous, reproducible, and mathematically formalized quantum machine learning algorithms.*
