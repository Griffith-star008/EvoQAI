# Reproducibility Statement

EvoQAI is committed to rigorous, transparent, and reproducible academic research. We reject the trend of overclaiming results in Quantum Machine Learning (QML) and provide all necessary tools for independent researchers to verify our findings.

## 1. Code Availability
The entire source code for EvoQAI is open-source and hosted on GitHub.
- **Repository:** https://github.com/Griffith-star008/EvoQAI

## 2. Dependency Management
All experiments were conducted using the exact library versions specified in `requirements.txt`.
Key libraries:
- `pennylane >= 0.33.0`
- `torch >= 2.0.0`

## 3. Determinism and Seeds
To ensure statistical validity, all comparative baselines (`experiments/run_baselines.py`) are executed across 3 fixed random seeds: `[42, 123, 999]`. Both the dataset generation (concept drift) and the model weight initializations are seeded.

## 4. One-Click Reproduction
Researchers can fully reproduce the plots and JSON metrics reported in our papers by running a single bash script from the root directory:
```bash
bash scripts/reproduce_all.sh
```

## 5. Hardware Assumptions
Current Phase 1 and Phase 2 experiments rely on `default.qubit` statevector simulators and simulated depth-induced depolarizing noise. Real QPU execution (Phase 3) is supported via `qiskit-ibm-provider`, but users must supply their own IBM Quantum API token to reproduce those specific hardware metrics.
