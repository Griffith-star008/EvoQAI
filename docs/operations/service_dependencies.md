# Service Dependency Mapping

The AQIP cluster is highly distributed. This document maps the critical and non-critical dependencies between microservices.

## 1. Dependency Graph
```mermaid
graph TD
    API[API Gateway] --> Engine[Runtime Engine]
    Engine --> QPU[QPU Controller (Hardware)]
    Engine --> GPU[GPU Worker (PyTorch)]
    
    Engine --> Policy[Policy Network Inference]
    Engine -.-> Twin[Digital Twin Simulator]
    Engine --> Verifier[SMT Verifier]
    
    Engine -.-> Telemetry[Prometheus Metrics]
    Engine -.-> Logger[Fluentd Log Aggregator]
```

## 2. Critical Path Dependencies (Hard Dependencies)
*If these services fail, execution halts immediately.*
- **API Gateway:** Required for job submission.
- **Runtime Engine:** The core DAG executor.
- **QPU / GPU Workers:** Required for actual computation.
- **SMT Verifier:** Required to guarantee safety. If the Verifier crashes, evolution stops and the system falls back to a static execution graph.

## 3. Non-Critical Dependencies (Soft Dependencies)
*If these services fail, execution continues with degraded performance or observability.*
- **Digital Twin Simulator:** If it fails, the Policy Network proposes permutations "blind", relying entirely on the SMT verifier for safety and live feedback for loss calculation (high latency penalty).
- **Prometheus / Fluentd:** If observability fails, the system continues executing, but evolution pauses because new telemetry data cannot be ingested.
- **Policy Network Inference:** If the meta-learned policy server goes offline, the system falls back to randomized heuristic graph permutations (extremely inefficient, but mathematically safe).
