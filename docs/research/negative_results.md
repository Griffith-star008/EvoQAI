# Negative Results and Failure Case Analysis

In the pursuit of scientific transparency, we document conditions under which the Autonomous Quantum Intelligence Platform (AQIP) performs worse than static baselines.

## 1. High-Noise Quantum Regimes
**Observation:** When physical quantum gate error rates exceed $p = 10^{-2}$ per 2-qubit gate, the SMT verifier struggles to differentiate between logical state divergence and physical noise collapse.
**Result:** The Digital Twin incorrectly simulates high expected loss, causing the Policy Network to reject theoretically optimal graph permutations.
**Conclusion:** AQIP requires fault-tolerant qubits or physical error rates $< 10^{-3}$ to successfully optimize deep hybrid circuits.

## 2. Ultra-Short Lifespan Workloads
**Observation:** For purely classical workloads taking less than 50ms to execute, the AQIP self-evolving loop introduces net-negative latency overhead.
**Result:** The time taken to profile, propose, verify, and deploy a permutation ($\sim 45$ ms) dominates the execution time.
**Conclusion:** AQIP should be disabled for trivial, non-iterative tasks. It is designed for long-running, iterative (e.g., VQE, RL) or highly concurrent distributed workflows.

## 3. Highly Entangled Subgraph Verification
**Observation:** When proposing a permutation that spans more than $w_{max}=20$ strongly entangled qubits, the SMT solver (`z3-solver`) encounters state-space explosion.
**Result:** The verifier reaches the timeout threshold ($\tau=100$ ms) in $>95\%$ of cases, resulting in conservative rejection of the permutation.
**Conclusion:** Bounded verification is strictly limited to localized subgraph rewrites. Global quantum architecture searches (QAS) cannot be verified in real-time.

## 4. Policy Network Catastrophic Forgetting
**Observation:** If the hardware topology changes radically at runtime (e.g., swapping a QPU for a simulated QVM mid-execution), the meta-learned policy network proposes extremely poor permutations for the first 10-15 cycles.
**Result:** The system relies entirely on SMT rejection to stay safe, wasting computational resources.
**Conclusion:** The meta-learning context window does not generalize zero-shot to entirely novel hardware constraints.
