# Threats to Validity

This document outlines potential threats to the validity of the claims and empirical results presented in the AQIP research.

## 1. Construct Validity
*Are we measuring what we claim to measure?*
- **The AQII Metric:** The Autonomous Quantum Intelligence Index (AQII) is a composite metric combining task accuracy, latency, and noise. Its weighting ($\alpha, \beta, \gamma$) is somewhat arbitrary. A different weighting scheme could make baselines appear more competitive. (Mitigated by Sensitivity Analysis).
- **Simulated Hardware:** While classical metrics were gathered on physical hardware (AMD EPYC, NVIDIA A100), quantum metrics heavily rely on IBM Qiskit Aer simulators with realistic noise models, rather than exclusively physical QPU runs, due to queue times.

## 2. Internal Validity
*Are there confounding factors affecting the results?*
- **Caching Effects:** Static baselines (MLIR, Qiskit Transpiler) might suffer from cold-start compilation times in our benchmarks, whereas AQIP's runtime evolution amortizes this cost.
- **SMT Solver Non-Determinism:** The `z3-solver` exhibits heuristic behavior. Varying random seeds in the solver might occasionally allow a timeout to resolve into a SAT/UNSAT result, adding variance to the evolution latency.

## 3. External Validity (Generalizability)
*Do the results generalize outside the specific benchmark conditions?*
- **Workload Bias:** The benchmarks (VQE, QML) are heavily biased towards variational algorithms. AQIP's benefits for non-variational quantum algorithms (e.g., Shor's) are unproven and likely minimal.
- **Hardware Architecture:** The platform was validated on a standard CPU+GPU+QPU topology. Generalization to analog quantum simulators (e.g., neutral atoms) or continuous-variable photonics is unsupported.

## 4. Conclusion Validity
*Are the statistical conclusions sound?*
- **Sample Size:** $N=50$ episodes per condition is statistically sufficient to detect large effect sizes (Cohen's $d > 0.8$), but may be underpowered to detect subtle degradations in edge cases.
- **Distribution Assumptions:** We use ANOVA which assumes normality. If AQII scores follow a highly skewed distribution (e.g., due to rare catastrophic SMT timeouts), non-parametric tests (Kruskal-Wallis) would be more robust.
