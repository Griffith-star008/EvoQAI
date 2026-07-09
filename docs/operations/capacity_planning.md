# Capacity Planning & Cost Optimization

Running AQIP in a production cloud environment requires careful orchestration of heterogeneous compute resources (CPUs, GPUs, QPUs).

## 1. Resource Footprint per Component

| Component | CPU Cores | Memory | Acceleration | Note |
|:---|:---:|:---:|:---:|:---|
| **UAIR Compiler** | 2 | 4 GB | None | Lightweight, heavily cached. |
| **SMT Verifier** | 8 | 16 GB | None | Heavily multi-threaded CPU bound. |
| **Policy Network**| 4 | 8 GB | GPU (T4/A10G) | Fast inference, small VRAM footprint. |
| **Digital Twin** | 16 | 32 GB | GPU (A100) | Requires high memory bandwidth. |

## 2. Scaling Ratios
For a balanced cluster avoiding bottlenecks:
- For every **1 Digital Twin node**, deploy **3 SMT Verifier nodes**.
- The Verifier is the critical path; under-provisioning CPU cores here will violate the 95% SLI for verification reliability.

## 3. Cost Optimization Strategies
- **Spot Instances:** The Digital Twin and Policy Network are stateless between cycles. Run them on Spot/Preemptible instances to reduce costs by up to 70%.
- **QPU Time-sharing:** Physical Quantum Processing Unit (QPU) access is astronomically expensive. Use the `qiskit-aer` GPU simulator for all intermediate verification. Only dispatch to physical QPUs when the graph converges to a local optimum.
- **Auto-scaling:** Bind the Kubernetes HPA (Horizontal Pod Autoscaler) to the custom Prometheus metric `aqip_verifier_queue_depth`. Scale out when queue > 10.
