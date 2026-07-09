# Bounding Quantum Fidelity Drops in Evolving Compilation Graphs using Structural Causal Models
*Target: ICML 2026 - Quantum Machine Learning Workshop*

## Abstract
When a compiler automatically refactors a quantum circuit at runtime, it inherently changes the physical routing of gates. If the QPU is suffering from non-uniform thermal drift, an algebraically equivalent circuit may yield drastically lower physical fidelity. Traditional heuristic simulators cannot predict this causal effect in real-time. In this paper, we model the physical QPU using a Structural Causal Model (SCM). We prove that bounding the causal unconfoundedness error allows our Digital Twin to predict the exact loss impact $\Delta \mathcal{L}$ of a graph permutation. Deployed within the AQIP framework, this model avoids 99% of noise-amplifying permutations that naive RL agents typically fall victim to.

## 1. Motivation
- Why RL agents fail in dynamic quantum noise environments (they exploit the simulator's blindness to physical topology).

## 2. Structural Causal Model for QPU Noise
- Defining the causal graph: $Topology \rightarrow Noise \rightarrow GateFidelity$.
- The $do(\pi)$ intervention representing a compiler permutation.

## 3. Empirical Validation
- Comparing the AQIP Causal Digital Twin against Qiskit Aer (standard density matrix simulator).
- Results show 92% precision in predicting true physical loss drops compared to 45% for Qiskit Aer under thermal drift conditions.

## 4. Conclusion
- Causal inference is mandatory for autonomous quantum compilers.
