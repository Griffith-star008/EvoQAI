# Causal Digital Twin-guided Safe Evolution of Hybrid Quantum-Classical Models

**Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
**Authors:** Huy Ngo Anh

## Abstract
Dynamic structural evolution of Variational Quantum Circuits (VQCs) is a powerful technique for addressing concept drift. However, on current Noisy Intermediate-Scale Quantum (NISQ) hardware, naively expanding circuit depth exponentially amplifies physical noise (e.g., depolarizing and thermal relaxation), eventually causing catastrophic model collapse. In this paper, we introduce a Causal Digital Twin (CDT) framework that acts as a safety gatekeeper for structural quantum evolution. By modeling the structural causal relationship between circuit depth, hardware noise, and output fidelity, the CDT preemptively evaluates proposed circuit mutations via *do-calculus*. If a mutation is predicted to violate safety thresholds, the CDT blocks the structural expansion and forces the system into a classical parameter-adaptation mode. Experimental simulations incorporating depth-induced depolarizing noise demonstrate that CDT-guided evolution prevents noise-induced accuracy collapse, maintaining robust performance where naive evolutionary strategies fail.

## 1. Introduction
- **Motivation:** NISQ devices have strictly limited coherence times. While algorithmic evolution (adding layers) increases theoretical expressivity, it physically incurs exponential noise penalties. 
- **Problem Statement:** Standard evolutionary algorithms optimize for theoretical loss reduction blindly, without understanding physical hardware constraints.
- **Contributions:**
  1. We formulate the first Structural Causal Model (SCM) linking algorithmic depth to NISQ hardware noise.
  2. We propose a Safe Evolution algorithm that queries a Causal Digital Twin using Judea Pearl's *do-calculus* before deploying structural mutations.
  3. We demonstrate through rigorous simulation that CDT intervention prevents catastrophic collapse under streaming concept drift.

## 2. Structural Causal Model for NISQ Evolution
### 2.1 The Causal Graph
Let the system be defined by a directed acyclic graph (DAG) $\mathcal{G}$:
$$ L \rightarrow N \rightarrow F $$
Where:
- $L$ (Depth): The number of parameterized entangling layers.
- $N$ (Noise Channel): The aggregate physical noise (e.g., depolarizing probability $p$).
- $F$ (Fidelity): The observable measurement fidelity of the quantum state.

### 2.2 Formalizing the Intervention (Do-Calculus)
During a concept drift event, the evolutionary engine proposes adding a layer. In causal inference terms, this is an intervention on the structural depth: $do(L = l + 1)$.
We wish to estimate the expected fidelity under this intervention:
$$ \mathbb{E}[F \mid do(L = l+1)] $$

Assuming a global depolarizing channel $\mathcal{E}$ parameterized by single-layer error rate $p$, the expectation value of an observable $O$ (e.g., Pauli-Z) decays as:
$$ \mathbb{E}[O_{noisy} \mid do(L)] = (1 - p)^L \cdot \mathbb{E}[O_{ideal}] $$

The Causal Digital Twin computes this counterfactual fidelity.

### 2.3 Safe Evolutionary Protocol
Let $\tau_{safe}$ be the threshold fidelity required to maintain a signal-to-noise ratio sufficient for classification.
The Causal Digital Twin (CDT) acts as a safety gate:
1. Engine proposes mutation $do(L = l + 1)$.
2. CDT evaluates: Is $\mathbb{E}[F \mid do(L = l+1)] \geq \tau_{safe}$?
3. **If Safe:** Execute quantum structural mutation.
4. **If Unsafe:** Reject quantum mutation. Fall back to classical adaptation (e.g., decaying learning rate $\alpha$ to stabilize existing weights).

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
