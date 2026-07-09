# Mathematical Appendix

## A. Proof Dependency Graph

The logical dependency structure of all formal results:

```
Assumption 1 (DAG) ──────────→ Lemma 1 (Topological Ordering)
                                    │
Assumption 2 (Bounded w) ────→ Lemma 2 (Verification Complexity)
                                    │
Assumption 3 (Monotonic) ────→ Proposition 2 (Termination)
Assumption 4 (Stationarity)       │
Assumption 5 (Finite Π) ─────→ Theorem 1 (Convergence)
                                    │
Lemma 3 (Boundedness) ───────→─┘
                                    
Lemma 2 ─────────────────────→ Theorem 2 (Complexity)
                                    │
                               Corollary 2 (Real-Time)

Proposition 3 (Soundness) ───→ Theorem 3 (Correctness)
                                    │
                               Corollary 1 (Stability)

Theorem 1 + Definition 6 ───→ Corollary 3 (AQII Improvement)
```

---

## B. SMT Encoding Details

### B.1 Classical Subgraph Encoding
For a classical subgraph consisting of arithmetic operations over 32-bit integers:
- Each variable is encoded as a bitvector of width 32.
- Each operation (`add`, `mul`, `shift`) is encoded as the corresponding SMT bitvector operation.
- The non-equivalence query checks whether any input bitvector assignment produces different outputs.

### B.2 Quantum Subgraph Encoding
For a quantum subgraph with $w$ qubits:
- The quantum state is encoded as a $2^w$-dimensional complex vector.
- Each gate (RX, RY, RZ, CNOT) is encoded as its unitary matrix.
- The output distribution is the squared magnitude of the final state vector.
- Equivalence is checked by comparing output distributions over all $2^w$ basis states.

### B.3 Hybrid Subgraph Encoding
For subgraphs containing both classical and quantum nodes:
- Classical inputs to parameterized quantum gates are treated as symbolic bitvectors.
- The quantum state is parameterized by these symbolic inputs.
- The SMT solver explores both the classical and quantum dimensions simultaneously.

---

## C. Global Loss Function Component Details

| Component | Symbol | Definition | Typical Weight |
|:---|:---|:---|:---|
| Task Loss | $\mathcal{L}_{task}$ | Application-specific objective (e.g., cross-entropy) | $\alpha = 0.30$ |
| Latency | $\mathcal{L}_{latency}$ | End-to-end execution time | $\beta = 0.20$ |
| Memory | $\mathcal{L}_{memory}$ | Peak memory consumption | $\gamma = 0.15$ |
| Energy | $\mathcal{L}_{energy}$ | Total power consumption (watts·seconds) | $\delta = 0.10$ |
| Security | $\mathcal{L}_{security}$ | Verified invariant violation count | $\varepsilon = 0.05$ |
| Communication | $\mathcal{L}_{comm}$ | Inter-node data transfer volume | $\zeta = 0.08$ |
| Reliability | $\mathcal{L}_{reliability}$ | 1 − success rate | $\eta = 0.05$ |
| Hardware Fit | $\mathcal{L}_{hardware}$ | Device utilization gap (1 − utilization) | $\theta = 0.05$ |
| Noise | $\mathcal{L}_{noise}$ | Quantum gate error accumulation | $\iota = 0.02$ |

---

## D. Benchmark Hardware Specifications

| Component | Specification |
|:---|:---|
| CPU | AMD EPYC 7763 (64 cores, 2.45 GHz) |
| GPU | NVIDIA A100 80GB SXM |
| QPU | IBM Eagle r3 (127 qubits, $T_1$ ≈ 300 μs) |
| Memory | 512 GB DDR4-3200 ECC |
| Network | InfiniBand HDR 200 Gb/s |
| Storage | NVMe SSD 3.84 TB |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11.9 |

---

## E. Statistical Test Reference

| Test | When Used | Implementation |
|:---|:---|:---|
| One-way ANOVA | Comparing ≥ 2 group means | `statistical_validator.py` |
| Cohen's d | Measuring effect size between 2 groups | `statistical_validator.py` |
| 95% Confidence Interval | Reporting precision of mean estimates | `statistical_validator.py` |
| Shapiro-Wilk | Testing normality assumption | Planned |
| Kruskal-Wallis | Non-parametric alternative to ANOVA | Planned |