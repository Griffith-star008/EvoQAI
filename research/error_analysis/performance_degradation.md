# Performance Degradation & Unstable Scenarios

This document analyzes edge cases where the AQIP framework remains functional but suffers significant performance degradation (AQII score drops below the static baseline).

## 1. Zero-Shot Topology Swaps
**Scenario:** A workload executing on an IBM Eagle (127Q) QPU is suddenly migrated to an IBM Osprey (433Q) QPU due to spot-instance preemption.
**Observation:** The Meta-Learned Policy Network has optimized the graph specifically for the heavy-hex topology of the Eagle processor. When migrated, the proposed permutations generate massive SWAP gate overhead on the Osprey processor.
**Degradation:** AQII score drops by $\sim 45\%$ for the first 20 evolution cycles until the Policy Network accumulates enough new reward signal to adapt to the new topology.
**Mitigation:** Re-initialize the Policy Network's context window whenever a `TopologyChange` event is detected.

## 2. Micro-Task Overhead
**Scenario:** A user submits a batch of 10,000 extremely small, independent UAIR graphs (e.g., executing a single quantum Hadamard gate and measuring).
**Observation:** The evolution engine takes $\sim 45$ ms to process each graph, while the execution itself takes $< 1$ ms.
**Degradation:** The total throughput of the system is throttled by the evolution loop, performing $10\times$ slower than a static queue.
**Mitigation:** Introduce a `minimum_latency_threshold`. Graphs executing faster than 10ms bypass the evolution engine entirely.

## 3. Digital Twin Divergence (Noise Drift)
**Scenario:** Over a 4-hour execution window, the physical QPU experiences thermal drift, causing the baseline two-qubit gate error rate to double.
**Observation:** The Digital Twin simulator is not updated with real-time calibration data, so it continues to predict outcomes based on the old noise model.
**Degradation:** The Twin predicts $\Delta \mathcal{L} < 0$ for permutations that actually increase physical noise, causing the system to deploy sub-optimal structures.
**Mitigation:** Force a Digital Twin re-calibration every 30 minutes by querying the QPU's calibration API.
