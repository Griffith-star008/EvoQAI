# Causal Digital Twin-guided Safe Evolution of Hybrid Quantum-Classical Models

**Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
**Authors:** Huy Ngo Anh

## Abstract
Dynamic structural evolution of Variational Quantum Circuits (VQCs) is a powerful technique for addressing concept drift. However, on current Noisy Intermediate-Scale Quantum (NISQ) hardware, naively expanding circuit depth exponentially amplifies physical noise (e.g., depolarizing and thermal relaxation), eventually causing catastrophic model collapse. In this paper, we introduce a Causal Digital Twin (CDT) framework that acts as a safety gatekeeper for structural quantum evolution. By modeling the structural causal relationship between circuit depth, hardware noise, and output fidelity, the CDT preemptively evaluates proposed circuit mutations via *do-calculus*. If a mutation is predicted to violate safety thresholds, the CDT blocks the structural expansion and forces the system into a classical parameter-adaptation mode. Experimental simulations incorporating depth-induced depolarizing noise demonstrate that CDT-guided evolution prevents noise-induced accuracy collapse, maintaining robust performance where naive evolutionary strategies fail.

## 1. Introduction
- **Motivation:** NISQ devices have strictly limited coherence times. While algorithmic evolution (adding layers) increases theoretical expressivity, it physically incurs exponential noise penalties. 
- **Problem Statement:** Standard reinforcement learning or evolutionary algorithms operate blindly; they optimize for theoretical loss reduction without understanding physical hardware constraints.
- **Contributions:**
  1. We formulate the first Structural Causal Model (SCM) for NISQ hardware noise and algorithmic depth.
  2. We propose a Safe Evolution algorithm that queries a Causal Digital Twin before deploying structural mutations.
  3. We demonstrate through rigorous simulation that CDT intervention prevents catastrophic collapse under streaming concept drift.

## 2. Methodology
### 2.1 The Structural Causal Model (SCM)
We define the system as a directed acyclic graph (DAG):
$L \rightarrow N \rightarrow F$
Where $L$ is the circuit layer count, $N$ is the physical noise channel (depolarization probability $p$), and $F$ is the observable fidelity.

### 2.2 Fidelity Decay Equation
For a global depolarizing channel, the expectation value decays as:
$\mathbb{E}[Z_{noisy}] = (1 - p)^L \cdot \mathbb{E}[Z_{ideal}]$
The Causal Digital Twin computes the predicted fidelity for a proposed mutation $do(L = l_{current} + 1)$.

### 2.3 Safe Evolutionary Protocol
When a concept drift is detected, the Engine requests a mutation. The CDT evaluates the mutation. If the predicted fidelity $(1-p)^{L+1} < \tau_{safe}$, the mutation is rejected (labeled `UNSAFE`). The Engine falls back to adjusting the classical learning rate (decaying $\alpha$ to stabilize learning) rather than altering the quantum topology.

## 3. Experimental Evaluation
### 3.1 Setup
- **Dataset:** SEA Stream Generator with dual sudden drifts at Epochs 50 and 100.
- **Noise Model:** Depth-induced depolarizing noise simulated within a PennyLane differentiable workflow ($p = 0.15$, $\tau_{safe} = 0.55$).

### 3.2 Results
*(Refer to `experiments/reports/causal_twin_adaptation.png`)*
We compared Unsafe (Naive) Evolution against CDT-Guided Safe Evolution. 
1. **Unsafe Evolution:** At Epoch 50, drift occurs. The naive engine adds a layer ($L=2$). At Epoch 100, drift occurs again, and the engine adds another layer ($L=3$). The compounded noise suppresses the signal, causing accuracy to collapse to random guessing.
2. **Safe Evolution:** At Epoch 50, the first layer is added safely. At Epoch 100, the CDT intercepts the second mutation, determining it unsafe. The engine decays the learning rate instead. The model successfully preserves signal integrity and avoids collapse.

## 4. Conclusion
Integrating Causal Inference with Quantum Machine Learning provides a necessary theoretical bound for algorithmic evolution on physical hardware. The Causal Digital Twin guarantees safe operational bounds, representing a critical step toward deploying autonomous QML in production AIoT systems.
