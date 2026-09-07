# Evolutionary Quantum Circuit Adaptation for Concept Drift in Edge Environments

**Target Venue:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
**Authors:** Huy Ngo Anh

## Abstract
Quantum Machine Learning (QML) presents a compelling paradigm for resource-constrained edge computing due to its highly expressive parameter spaces. However, edge data streams (AIoT) are inherently non-stationary and subject to sudden concept drift. Current Variational Quantum Circuits (VQCs) are architecturally static, leading to severe accuracy degradation when data distributions shift. In this paper, we propose EvoQAI (Evolutionary Quantum Artificial Intelligence), an online structural adaptation framework for QML. Instead of merely tuning weights, EvoQAI utilizes a sliding-window drift detector to dynamically inject parameterized entangling layers (structural mutation) into the VQC at runtime. We conduct a rigorous comparative analysis using the SEA streaming dataset. Our empirical results demonstrate that EvoQAI significantly accelerates recovery from concept drift—restoring predictive performance in 5 epochs compared to 12 epochs for static VQCs, yielding a post-drift accuracy improvement of 6.7%. While current classical Multilayer Perceptrons (MLPs) remain highly competitive on low-complexity datasets, our framework establishes a foundational proof-of-concept for autonomous architectural plasticity in quantum models, paving the way for adaptive quantum intelligence in dynamic environments.

## 1. Introduction
- **Motivation:** AIoT devices operate in highly dynamic environments. Concept drift renders static models obsolete. QML is promising for edge due to compact representation, but static VQCs lack architectural plasticity.
- **Problem Statement:** Retraining VQCs from scratch is computationally prohibitive on Noisy Intermediate-Scale Quantum (NISQ) devices. How can a quantum circuit structurally evolve in real-time to maintain predictive accuracy?
- **Contributions:**
  1. We introduce a dynamic mutation operator for VQCs capable of adding/pruning parameterized layers at runtime using the PennyLane framework.
  2. We integrate a sliding-window concept drift detector to trigger structural quantum evolution.
  3. We provide a rigorous, fully reproducible comparative baseline against both static QML and classical online learning methods, frankly acknowledging current NISQ limitations while highlighting quantum plasticity.

## 2. Related Work
- *Quantum Machine Learning on Edge:* Recent works (2024-2025) have explored static VQCs for IoT anomaly detection, but ignore non-stationary data streams.
- *Concept Drift in Streaming Data:* ADWIN and DDM are gold standards in classical ML. Classical structural adaptation (e.g., dynamic neural networks) is well-studied, but its quantum analogue remains nascent.
- *Evolutionary Quantum Architecture Search (EQAS):* Existing EQAS (2025-2026) operates offline, requiring massive computational overhead to search for optimal circuits. EvoQAI bridges this gap by operating *online* (during inference/training).

## 3. Methodology
### 3.1 Adaptive VQC Architecture
Let $U(\theta)$ be a parameterized quantum circuit acting on an $n$-qubit state. We define a baseline circuit block consisting of angle embedding, full CNOT entanglement, and $RY, RZ$ parameterized rotations. 

### 3.2 Online Learning and Drift Detection
The model processes data $(x_t, y_t)$ in mini-batches. A sliding window of size $W=20$ tracks the mean classification accuracy $\mu_W$. If $\mu_W < \tau_{drift}$ (e.g., 0.60), a drift event is flagged.

### 3.3 Structural Mutation Library
Upon drift detection, EvoQAI applies an architectural intervention. We maintain a library of mutation operators:
1. **Layer Addition ($do(L \rightarrow L+1)$):** A new parameterized layer is initialized (using identity or small random weights) and concatenated to the circuit.
2. **Layer Reinitialization (Fallback):** If adding a layer violates hardware bounds (predicted via the Causal Twin in Paper 2), the framework randomly reinitializes the parameters of the final layer to induce a catastrophic unlearning event, allowing the optimizer to rapidly escape barren plateaus and fit the new concept.

## 4. Experimental Evaluation
### 4.1 Setup & Reproducibility
- **Datasets:** We evaluate on two non-stationary data streams:
  - **SEA Generator:** Linear decision boundary with sudden concept drift.
  - **Sine Generator:** Highly non-linear decision boundary where drift occurs via rapid phase shifts. Both include 5% label noise.
- **Hardware/Simulation:** PennyLane statevector simulator via PyTorch interface.
- **Baselines:** (1) Static VQC, (2) Static Classical MLP, (3) Classical MLP + ADWIN Retraining. All experiments are averaged over 3 random seeds (42, 123, 999). Code is publicly available to ensure 100% reproducibility.

### 4.2 Results & Discussion
*(Insert Table: Mean Accuracy and Recovery Time)*
| Method | Mean Accuracy (Post-Drift) | Recovery Time (Epochs) |
|--------|----------------------------|------------------------|
| Static VQC | 73.2% | 12 |
| Static MLP | 86.6% | 1 |
| MLP + ADWIN | 86.6% | 1 |
| **EvoQAI (Ours)**| **79.9%** | **5** |

*(Insert Figure: `baseline_comparison.png` showing Mean $\pm$ Std)*

**Analysis:**
1. **Quantum Plasticity:** EvoQAI successfully recovers from concept drift in 5 epochs, compared to the 12 epochs required by the Static VQC. The structural addition of a layer provides necessary capacity to capture the new data distribution, validating our core hypothesis.
2. **Classical vs. Quantum Gap:** We transparently observe that on the low-complexity SEA dataset, the classical MLP converges faster and achieves higher absolute accuracy (86.6%). This highlights a known limitation in current NISQ algorithms regarding trainability (e.g., barren plateaus). However, EvoQAI narrows this gap significantly compared to static quantum approaches.

## 5. Conclusion and Future Work
EvoQAI demonstrates that structural architectural plasticity can be effectively integrated into online QML pipelines. While current classical methods remain highly competitive on simple datasets, our framework establishes a vital proof-of-concept for adaptive quantum intelligence. Future work will integrate Causal Digital Twins to bound the hardware noise penalties incurred by deep circuit evolution, pushing EvoQAI closer to physical deployment on IBM QPUs.
