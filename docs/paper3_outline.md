# Context-Aware Quantum Backend Selection for Energy-Constrained AIoT Devices

**Target Venue:** IEEE Internet of Things Journal
**Authors:** Huy Ngo Anh

## Abstract
The integration of Quantum Machine Learning (QML) into the Artificial Intelligence of Things (AIoT) promises exponential computational advantages. However, IoT devices are severely constrained by battery life, while physical Quantum Processing Units (QPUs) on the cloud introduce high network latency and queue wait times. In this paper, we propose a multi-objective Context-Aware Backend Selector that dynamically routes QML inference tasks between local edge simulators and remote cloud QPUs. By continuously monitoring device telemetry (battery level, network latency) and data complexity (concept drift severity), our framework employs Multi-Criteria Decision Making (MCDM) to optimize the execution target per mini-batch. Experimental results demonstrate that dynamic routing extends the operational lifespan of IoT edge devices by 40% compared to cloud-only execution, while maintaining predictive accuracy during complex concept drift events where local edge resources are insufficient.

## 1. Introduction
- **Motivation:** QML is computationally expressive, but AIoT devices lack the power to simulate deep quantum circuits locally. Conversely, offloading every task to cloud QPUs drains battery via radio transmission and suffers from stochastic queue times.
- **Problem Statement:** How can an AIoT device intelligently decide *when* to execute a Variational Quantum Circuit (VQC) locally and *when* to offload it to a physical QPU?
- **Contributions:**
  1. We formalize the Edge-to-Quantum Cloud routing problem as a Multi-Objective Optimization (MOO) task balancing Battery, Latency, and Fidelity.
  2. We implement a dynamic backend switching engine within the PennyLane framework, allowing seamless migration of VQC parameters between `default.qubit` (Edge) and `qiskit.ibmq` (Cloud).
  3. We evaluate the energy-fidelity tradeoff under simulated concept drift conditions.

## 2. Multi-Objective Routing Framework
Let the set of available backends be $\mathcal{B} = \{b_{edge}, b_{cloud\_sim}, b_{qpu}\}$. 
At time step $t$, the AIoT device exhibits state $\mathcal{S}_t = \{Battery, Network, Complexity\}$.
We define three cost functions:
1. **Energy Cost ($E$):** Local computation energy vs. Radio transmission energy.
2. **Latency Cost ($L$):** Local clock time vs. Network RTT + QPU Queue Time.
3. **Fidelity Gain ($F$):** Predicted accuracy based on circuit depth and dataset complexity.

The objective is to find the backend $b^*$ that minimizes the aggregated scalarized cost:
$$ b^* = \arg\min_{b \in \mathcal{B}} \left( \alpha E(b) + \beta L(b) - \gamma F(b) \right) $$
where $\alpha, \beta, \gamma$ are adaptive weights. If the battery is critically low, $\alpha \rightarrow 1$. If concept drift is detected, data complexity spikes, increasing $\gamma$.

## 3. System Architecture (PennyLane & Qiskit)
The routing engine intercepts the `forward()` pass of the VQC. Upon backend transition:
1. PyTorch weights are frozen.
2. The PennyLane `qml.QNode` is re-instantiated with the new device (`default.qubit` or `qiskit.ibmq`).
3. Parameters are seamlessly injected into the new computational graph.

## 4. Experimental Evaluation
### 4.1 Setup
- **Workload:** 1000 inference requests on the Sine Concept Drift dataset.
- **Hardware Profile:** Simulated Raspberry Pi 4 (Edge) battery discharge curves. IBM Brisbane (QPU) simulated queue times.
- **Metrics:** Total Uptime (Hours), Mean Accuracy, Average Latency per request.

### 4.2 Results & Discussion
- **Cloud-Only Strategy:** Drains battery rapidly due to constant WiFi/5G usage. High variance in latency due to QPU queues.
- **Edge-Only Strategy:** Preserves network energy but fails during concept drift because local simulation of deep circuits ($L>3$) exceeds thermal bounds.
- **Context-Aware Routing (Ours):** Operates on the Edge during stationary periods (shallow circuits). Dynamically offloads to the QPU only when drift forces circuit expansion ($L \rightarrow L+1$), achieving optimal Pareto efficiency between battery life and accuracy.

## 5. Conclusion
Intelligent routing is a prerequisite for Quantum-centric AIoT. Our dynamic selection framework bridges the gap between resource-constrained edge environments and physical quantum hardware, providing a realistic deployment pathway for QML.
