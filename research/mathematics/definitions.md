# Formal Definitions

## Definition 1: UAIR Graph
A **Universal AI Intermediate Representation (UAIR) graph** is a directed acyclic graph $\mathcal{G} = \langle \mathcal{V}, \mathcal{E} \rangle$ where every node $v \in \mathcal{V}$ satisfies the Static Single Assignment (SSA) property: each variable is assigned exactly once and every use of a variable is dominated by its definition.

The node set is partitioned as $\mathcal{V} = V_{tensor} \cup V_{quantum} \cup V_{agent}$, where:
- $V_{tensor}$: Classical tensor operations (matmul, conv, softmax, etc.)
- $V_{quantum}$: Quantum gate operations (RX, RY, RZ, CNOT, Measure, etc.)
- $V_{agent}$: Agent decision nodes (plan, reflect, communicate, etc.)

## Definition 2: Structural Permutation
A **structural permutation** $\pi: \mathcal{G} \to \mathcal{G}'$ is a semantics-preserving graph rewrite that transforms the UAIR graph while maintaining output equivalence:
$$\pi(\mathcal{G}) = \mathcal{G}' \implies \forall x \in \text{dom}(\mathcal{G}),\ \text{Output}(\mathcal{G}, x) = \text{Output}(\mathcal{G}', x)$$

Examples include: Dead Node Elimination, Quantum Gate Fusion, Tensor Kernel Fusion, Operator Reordering.

## Definition 3: Global Loss Function
The **Global Loss Function** $\mathcal{L}: \mathcal{G} \times \mathcal{H} \to \mathbb{R}_{\geq 0}$ is defined as:
$$\mathcal{L}(\mathcal{G}, \mathcal{H}) = \alpha \mathcal{L}_{task} + \beta \mathcal{L}_{latency} + \gamma \mathcal{L}_{memory} + \delta \mathcal{L}_{energy} + \varepsilon \mathcal{L}_{security} + \zeta \mathcal{L}_{comm} + \eta \mathcal{L}_{reliability} + \theta \mathcal{L}_{hardware} + \iota \mathcal{L}_{noise}$$
where each $\mathcal{L}_i \geq 0$ and the weights satisfy $\alpha + \beta + \gamma + \delta + \varepsilon + \zeta + \eta + \theta + \iota = 1$.

## Definition 4: Runtime State
The **Runtime State** at timestep $t$ is the tuple:
$$s_t = (\mathcal{G}_t, \mathcal{H}_t, \mathcal{M}_t, \mathcal{P}_t, \mathcal{R}_t)$$
where $\mathcal{G}_t$ is the current UAIR graph, $\mathcal{H}_t$ is the hardware state, $\mathcal{M}_t$ is the memory state, $\mathcal{P}_t$ is the active scheduling policy, and $\mathcal{R}_t$ is the reflection log.

## Definition 5: Semantic Equivalence
Two UAIR graphs $\mathcal{G}$ and $\mathcal{G}'$ are **semantically equivalent**, written $\mathcal{G} \equiv \mathcal{G}'$, if and only if for all valid inputs $x$:
$$\text{Output}(\mathcal{G}, x) = \text{Output}(\mathcal{G}', x) \wedge \text{SideEffects}(\mathcal{G}, x) = \text{SideEffects}(\mathcal{G}', x)$$

## Definition 6: Autonomous Quantum Intelligence Index (AQII)
The **AQII** is a normalized composite score measuring the degree of autonomous intelligence:
$$\text{AQII} = \frac{1}{K} \sum_{k=1}^{K} w_k \cdot \frac{m_k - m_k^{baseline}}{m_k^{oracle} - m_k^{baseline}} \times 100$$
where $m_k$ is the measured value for metric $k$, and $w_k$ are importance weights.