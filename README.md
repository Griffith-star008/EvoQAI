# EvoQAI: Evolutionary Quantum Artificial Intelligence

EvoQAI is a rigorous Quantum Machine Learning (QML) research framework focused on **Evolutionary Quantum Circuit Adaptation under Concept Drift in Edge Environments**.

Unlike static VQC (Variational Quantum Circuit) models that fail when data distributions shift or quantum hardware noise drifts, EvoQAI actively monitors the streaming loss landscape and **structurally mutates** the quantum circuit at runtime (e.g., expanding circuit depth to increase capacity, or pruning gates to reduce noise).

## Core Research Focus (Targeting IEEE Tier-1)
1. **Evolutionary Circuit Adaptation:** Dynamic appending/pruning of parameterized quantum layers based on real-time drift metrics.
2. **Causal Digital Twin (WIP):** SCM-based bounding of noise interference during circuit mutation.

## Architecture
- **Quantum Core:** Implemented natively using **PennyLane** and **PyTorch**, supporting true adjoint differentiation and full statevector simulation.
- **Data Stream:** Custom SEA concept drift generator for streaming non-stationary data.
- **Engine:** PyTorch-based online learning loop with sliding-window drift detection.

## Getting Started

### Installation
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### Running the End-to-End Experiment
This script streams 150 batches of data, triggers a sudden concept drift at epoch 75, and demonstrates the VQC autonomously adding a quantum layer to recover its accuracy.
```bash
python run_experiments.py
```
The output plot will be saved to `experiments/reports/drift_adaptation.png`.

## Reproducibility
This repository is designed for rigorous academic reproducibility. All metrics, seeds, and hyperparameters are deterministic.

*Note: Previous claims regarding C++/CUDA bindings have been removed as the project transitions to a pure PennyLane/PyTorch differentiable stack to prioritize research velocity and algorithmic novelty.*
