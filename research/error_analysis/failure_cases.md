# Hard Failure Cases

This document records scenarios where the AQIP framework experiences catastrophic failure (system crash, invariant violation, or irrecoverable stall).

## 1. Z3 Solver OOM (Out of Memory)
**Context:** The evolution engine proposes a graph permutation involving a subgraph of width $w = 26$ qubits.
**Trigger:** The SMT constraint array for a $2^{26}$ state vector exceeds the available RAM (32GB) on the verifier node.
**Failure Mode:** The Linux OOM Killer terminates the `z3-solver` process. The Python runtime catches a `BrokenPipeError` but fails to recover the verification state.
**Workaround:** Strictly enforce the $w_{max} \leq 20$ budget in `performance_budgets.json`.

## 2. Distributed Graph Partitioning Deadlock
**Context:** The UAIR graph is partitioned across a CPU worker and a QPU controller.
**Trigger:** The network link drops packets during a highly synchronous hybrid loop (e.g., VQE gradient calculation), causing the QPU controller to wait indefinitely for classical parameters while the CPU worker waits for quantum expectation values.
**Failure Mode:** The entire execution DAG deadlocks.
**Workaround:** Implement aggressive timeouts (e.g., 500ms) on cross-domain RPC calls, falling back to a Checkpoint/Restore sequence.

## 3. Telemetry Poisoning
**Context:** A rogue process on a shared Kubernetes node artificially spikes CPU usage, causing the `MetricsCollector` to report extreme latency for a specific classical subgraph.
**Trigger:** The Policy Network assumes this subgraph is poorly optimized and begins proposing continuous permutations to "fix" it.
**Failure Mode:** Evolution thrashing. The system spends 100% of its time evolving and verifying, but execution time never improves because the bottleneck is external.
**Workaround:** Cross-validate node-level metrics (e.g., `node_cpu_utilization`) against graph-level metrics. Disable evolution if node utilization $> 95\%$.
