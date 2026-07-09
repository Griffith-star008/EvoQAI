# Evaluation Protocol

To ensure rigorous, unbiased, and reproducible empirical validation, all benchmarks MUST adhere to the following protocol.

## 1. Pre-Registration of Hypotheses
Before running any evaluation, the researcher must clearly state the hypothesis.
**Example:** *Hypothesis 1: Enabling cross-domain fusion will reduce hybrid memory overhead by at least 30% compared to disjoint MLIR/Qiskit compilation.*

## 2. Acceptance Criteria
The hypothesis is only considered "Accepted" if:
1. $p < 0.05$ (using one-way ANOVA).
2. Cohen's $d > 0.8$ (representing a "large" effect size).
3. The lower bound of the 95% Confidence Interval for the target metric meets the stated improvement threshold.

## 3. Environment Sanitization
Prior to execution:
1. All caches must be cleared.
2. Background processes on benchmark nodes must be suspended.
3. Network latency between CPU and QPU controllers must be measured and recorded.

## 4. Execution Rules
1. **Blinding:** The Digital Twin simulator must not have access to the ground-truth test labels during QML evaluations.
2. **Seeds:** All random seeds (Numpy, PyTorch, Qiskit) must be fixed and documented in `benchmark_config.json`.
3. **Repetitions:** Minimum $N=50$ independent trials per configuration.

## 5. Artifact Archival
Upon completion, the exact commit hash, `SBOM.json`, raw JSONL telemetry logs, and the statistical report must be zipped and cryptographically signed.
