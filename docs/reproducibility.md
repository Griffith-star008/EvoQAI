# Reproducibility & Artifact Evaluation Guide

## Commitment to Science
The AIOT Global Production Framework (AQIP) adheres to the highest standards of scientific reproducibility. This guide outlines how peer-reviewers and independent researchers can re-run the benchmark experiments and achieve the exact same statistical outputs for the Autonomous Quantum Intelligence Index (AQII) claimed in the dissertation.

## Environment Lock
The framework leverages strict deterministic seeding and version locking.
- **Python Version:** 3.11.x
- **Quantum Backends:** Qiskit Aer 0.13.0, Pennylane 0.35.0
- **Random Seeds:** By default, `ExperimentEngine` fixes the numpy and PyTorch seeds to `42` before initializing the UAIR graph.

## Artifact Reproduction Steps

### 1. Docker-based Execution
The most reliable method to reproduce the distributed benchmark is via the provided `docker-compose.yml`. This guarantees the identical OS and library environment.
```bash
# 1. Initialize the cluster (Master + Workers + Redis)
docker-compose up -d

# 2. Trigger the AQII Scientific Benchmark (N=20 episodes)
docker exec -it aqip-kernel-master python src/research/benchmark/benchmark_engine.py

# 3. Retrieve the Standardized Markdown Report
docker cp aqip-kernel-master:/app/research/benchmark/results/ ./local_results/
```

### 2. Statistical Validation Requirements
To reproduce the statistical confidence intervals (95% CI) and ANOVA tables reported in Chapter 8:
- Ensure the benchmark runs for at least $N=50$ episodes when running the `Category_K_Robustness` stress test.
- The `aqii_calculator.py` automatically normalizes the variance across distributed nodes.

## Expected Outcomes
If the environment is correctly replicated, the mean AQII score for the baseline static pipeline should be $11.00 \pm 1.2$, while the AQIP autonomous pipeline should stabilize at $86.07 \pm 2.4$, confirming the theoretical claims in Chapter 3.
