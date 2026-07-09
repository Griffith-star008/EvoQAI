# Lemmas

## Lemma 1: Topological Ordering Existence
**Statement:** For any valid UAIR graph $\mathcal{G} = \langle \mathcal{V}, \mathcal{E} \rangle$ satisfying Assumption 1 (DAG Structure), there exists at least one topological ordering $\sigma: \mathcal{V} \to \{1, 2, \ldots, |\mathcal{V}|\}$ such that for every edge $(u, v) \in \mathcal{E}$, $\sigma(u) < \sigma(v)$.

**Proof:** By Assumption 1, $\mathcal{G}$ contains no directed cycles. By Kahn's algorithm, we can iteratively remove nodes with zero in-degree. Since $\mathcal{G}$ is acyclic, at every step there exists at least one node with zero in-degree. The removal order defines a valid topological ordering. $\square$

**Significance:** This lemma guarantees that the UAIR compiler can always schedule operations in a valid execution order, which is a prerequisite for the complexity bounds in Theorem 1.

---

## Lemma 2: Bounded Verification Complexity
**Statement:** Given a structural permutation $\pi$ that affects a subgraph $\mathcal{G}_{sub} \subseteq \mathcal{G}$ with at most $w$ quantum nodes, the SMT verification of semantic equivalence $\mathcal{G}_{sub} \equiv \pi(\mathcal{G}_{sub})$ requires at most $\mathcal{O}(2^w \cdot |\mathcal{V}_{sub}|)$ satisfiability checks.

**Proof:** Each quantum node in $\mathcal{G}_{sub}$ contributes a 2-dimensional complex amplitude (qubit state). The joint state space of $w$ qubits is $2^w$-dimensional. For each basis state, verifying that the classical and quantum output distributions match requires a linear scan over $|\mathcal{V}_{sub}|$ nodes. Hence the total verification cost is $\mathcal{O}(2^w \cdot |\mathcal{V}_{sub}|)$. $\square$

**Significance:** Combined with Assumption 2 ($w \leq w_{max} = 20$), this bounds verification to a tractable constant factor, enabling real-time evolution.

---

## Lemma 3: Loss Boundedness
**Statement:** For any valid UAIR graph $\mathcal{G}$ and hardware state $\mathcal{H}$, $\mathcal{L}(\mathcal{G}, \mathcal{H}) \geq 0$.

**Proof:** Each component $\mathcal{L}_i \geq 0$ by construction (they measure non-negative quantities: latency, memory usage, noise magnitude, etc.). Since all scalarization weights $\alpha, \beta, \ldots, \iota \geq 0$, the weighted sum is non-negative. $\square$

**Significance:** This lower bound is critical for the convergence proof in Theorem 1.