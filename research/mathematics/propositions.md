# Propositions

## Proposition 1: UAIR Subsumes Existing IRs
**Statement:** For any program representable in MLIR's Tensor dialect or Qiskit's DAGCircuit, there exists a semantically equivalent UAIR graph $\mathcal{G}$.

**Argument:** MLIR's Tensor dialect operates on ranked tensor types with affine loop nests — these map directly to $V_{tensor}$ nodes with shaped-tensor edge annotations. Qiskit's DAGCircuit consists of quantum gate nodes with qubit-wire edges — these map directly to $V_{quantum}$ nodes with qubit-register edge annotations. Since UAIR's node set $\mathcal{V} = V_{tensor} \cup V_{quantum} \cup V_{agent}$ is a strict superset, any program expressible in either MLIR or DAGCircuit is expressible in UAIR. $\square$

**Significance:** This establishes UAIR as a strictly more expressive IR, justifying its claim as a "Universal" representation.

---

## Proposition 2: Monotonic Improvement Implies Termination
**Statement:** If the evolution sequence $\{\mathcal{G}_t\}$ satisfies $\mathcal{L}(\mathcal{G}_{t+1}) < \mathcal{L}(\mathcal{G}_t)$ for all $t$ where a permutation is deployed, and the set of reachable graphs is finite, then the evolution terminates.

**Argument:** Each deployment produces a strictly smaller loss value. Since $\mathcal{L} \geq 0$ (Lemma 3) and the number of distinct reachable graphs from $\mathcal{G}_0$ under $\Pi$ is finite (Assumption 5), the sequence cannot decrease indefinitely. $\square$

**Significance:** This proposition is the core building block for Theorem 1 (Convergence).

---

## Proposition 3: Verification Soundness Implies Evolution Safety
**Statement:** If the SMT solver is sound, then no deployed permutation can change the program's observable behavior.

**Argument:** Soundness means UNSAT $\implies$ the non-equivalence formula has no satisfying assignment $\implies$ no input exists that distinguishes $\mathcal{G}$ from $\pi(\mathcal{G})$. By Definition 5 (Semantic Equivalence), $\mathcal{G} \equiv \pi(\mathcal{G})$. $\square$

**Significance:** This proposition underpins the safety guarantee in Theorem 3.

---

## Proposition 4: Digital Twin Prediction Error is Bounded
**Statement:** If the Digital Twin model $\hat{\mathcal{L}}$ is trained on $N$ historical evolution samples with bounded noise $\sigma^2$, then the prediction error satisfies:
$$\mathbb{E}[|\hat{\mathcal{L}}(\mathcal{G}') - \mathcal{L}(\mathcal{G}')| ] \leq \mathcal{O}\left(\sqrt{\frac{\sigma^2 \cdot d}{N}}\right)$$
where $d$ is the feature dimension of the graph embedding.

**Argument:** By standard supervised learning generalization bounds (Rademacher complexity), a model trained on $N$ i.i.d. samples with bounded loss has expected test error decreasing as $\mathcal{O}(\sqrt{d/N})$. $\square$

**Significance:** As the system accumulates evolution history, the Digital Twin's predictions become increasingly accurate, leading to fewer wasted verification attempts.