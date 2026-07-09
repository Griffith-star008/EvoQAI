# RQ1: Universal AI Intermediate Representation (UAIR)

## Problem
Hybrid quantum-classical workflows suffer from extreme context-switching latencies because they rely on disjoint compilation toolchains (e.g., PyTorch to MLIR vs. Qiskit to OpenQASM). There is no single graph representation that natively supports both domains.

## Motivation
To eliminate serialization overhead and enable cross-domain compiler optimizations (like fusing classical parameters directly into quantum gates), the compiler must possess a unified view of the entire hybrid program.

## Hypothesis
If quantum gate operations, classical tensor operations, and agent decision nodes are mapped as peer nodes onto a singular Static Single Assignment (SSA) Directed Acyclic Graph (DAG), then hybrid scheduling overhead will be reduced by at least 30%, and cross-domain fusion rate will exceed 50%.

## Methodology
1. **Design:** Extend the MLIR SSA semantics to support `QuantumRegister` and `AgentMessage` types.
2. **Implementation:** Build the UAIR compiler frontend that parses PyTorch and Qiskit ASTs into the unified graph.
3. **Execution:** Implement an executor that topologically sorts the hybrid DAG and dispatches subgraphs to the respective hardware backends.

## Evaluation
- **Workloads:** Variational Quantum Eigensolver (VQE) on $H_2$, Quantum Neural Networks (QNN) on MNIST.
- **Metrics:** Hybrid scheduling overhead (ms), Cross-domain fusion rate (%), Peak memory consumption (MB).
- **Baselines:** Disjoint execution using standard Qiskit + PyTorch toolchains.

## Expected Outcome
The UAIR compiler will strictly dominate the baseline in latency and memory overhead by eliminating intermediate serialization and enabling joint classical-quantum graph rewrites.

## Threats to Validity
- **Construct Validity:** "Hybrid scheduling overhead" is difficult to isolate from hardware-specific dispatch latencies.
- **External Validity:** The unified graph assumes discrete quantum gate operations. It may not generalize to continuous-variable quantum photonics.
