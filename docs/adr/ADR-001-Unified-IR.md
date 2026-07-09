# ADR 001: Unified AI Intermediate Representation (UAIR)

**Status:** Accepted
**Date:** 2026-07-09

## Context and Problem Statement
Hybrid computational workflows combining Deep Learning (Tensors) and Quantum Circuits (Qubits) suffer from extreme context-switching latencies because they rely on disjoint compilation toolchains (e.g., PyTorch -> MLIR, Qiskit -> OpenQASM).

## Decision
We will construct the Universal AI Intermediate Representation (UAIR). UAIR will force all `QuantumGate`, `TensorOp`, and `AgentDecision` operations to exist as nodes on a single Static Single Assignment (SSA) Directed Acyclic Graph (DAG).

## Alternatives Considered
- **Plugin Bridge:** Building a translator between MLIR and QASM. Rejected because it maintains two separate memory spaces, preventing cross-domain compiler fusion.

## Consequences
- **Positive:** Enables unprecedented cross-domain optimization (e.g., fusing classical neural network weights directly into parameterized quantum rotations).
- **Negative:** Requires writing a custom graph compiler from scratch, rather than relying on off-the-shelf solutions.
