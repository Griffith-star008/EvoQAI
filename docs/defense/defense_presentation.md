# PhD Defense Presentation
*Title: Autonomous Quantum Intelligence Platform (AQIP)*
*Duration: 45 Minutes*

---
## Slide 1: Title & Introduction
- **Title:** Meta-Evolutionary QuantumIR Adaptation via Causal Digital Twin
- **Candidate:** Huy Ngo Anh
- **Goal:** Solving the hybrid quantum-classical scheduling bottleneck and the noise-drift compilation problem.

---
## Slide 2: The Problem (Serialization & Static Compilation)
- **Visual:** A diagram showing PyTorch halting, serializing arrays, sending them over RPC to Qiskit, Qiskit running, and returning arrays.
- **Key Point:** Current systems (Ray, PennyLane) treat the boundary as a black box.
- **Key Point:** Static compilation fails when QPU hardware noise drifts 30 minutes into execution.

---
## Slide 3: Core Contribution 1 - UAIR
- **Universal AI Intermediate Representation.**
- **Visual:** A single SSA DAG containing both $V_{tensor}$ (Matrix Multiply) and $V_{quantum}$ (CNOT).
- **Impact:** Eliminates the boundary. Allows cross-domain fusion.

---
## Slide 4: Core Contribution 2 - Meta-Evolution
- **The Concept:** The compiler shouldn't stop at run-time. It should evolve the DAG.
- **The Engine:** Telemetry $\to$ Policy Network $\to$ Propose $\pi(\mathcal{G})$.
- **The Causal Twin:** We don't test on the real QPU (too slow). We test on a Structural Causal Model (SCM) to predict true fidelity drop.

---
## Slide 5: Core Contribution 3 - Zero-Trust Verification
- **The Risk:** Neural networks hallucinate. A bad graph permutation breaks the math.
- **The Defense:** We isolate the subgraph ($w_{max} \le 20$) and run Z3 SMT equivalence checking.
- **Result:** $<100$ms latency, 0% False Acceptance Rate.

---
## Slide 6: Experimental Results (Scale & Ablation)
- **Large-Scale Simulation:** Show the 10,000 node graph. 10.7x speedup over Ray.
- **Ablation Study:** Show the 666% improvement in AQII score vs static baseline.

---
## Slide 7: Conclusion & Publications
- **Publications:** NeurIPS 2026 (Under Review), ICML QML Workshop.
- **Final Thought:** AQIP elevates autonomous compilation to an enterprise-grade, mathematically safe reality.

---
## Slide 8: Q&A
- *Backup slides for Causal Bounds (Theorem 3.2) and SMT Complexity (Theorem 2).*
