# AQIP API Examples

This document demonstrates how to interact with the stable AQIP APIs (`aqip.core.*`).

## 1. Compiling a Hybrid Program

```python
import torch
import qiskit
from aqip.core import compiler, runtime

# Define classical logic
class ClassicalNet(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

# Define quantum logic
qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Compile into UAIR DAG
uair_graph = compiler.build_hybrid_graph(
    classical_module=ClassicalNet(),
    quantum_circuit=qc,
    fusion_rules="aggressive"
)
```

## 2. Starting the Autonomous Runtime

```python
from aqip.core.engine import SelfEvolvingRuntime
from aqip.core.objectives import GlobalLoss

# Define custom objective weights
loss = GlobalLoss(
    task_weight=0.5,
    latency_weight=0.3,
    noise_weight=0.2
)

# Initialize and run
runtime = SelfEvolvingRuntime(
    graph=uair_graph,
    loss_function=loss,
    enable_smt_verification=True
)

optimized_graph = runtime.evolve(max_cycles=100, timeout_ms=5000)
```

## 3. Querying Telemetry

```python
from aqip.core.observability import MetricsClient

client = MetricsClient(endpoint="http://localhost:9090")

# Fetch real-time AQII score
aqii = client.get_gauge("aqip_aqii_score")
print(f"Current AQII: {aqii}")

# Fetch evolution history
history = client.get_histogram("evolution_latency_ms")
```
