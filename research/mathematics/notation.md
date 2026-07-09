# Mathematical Notation & Symbol Reference

## Sets and Spaces

| Symbol | Definition |
|:---|:---|
| $\mathcal{G} = \langle \mathcal{V}, \mathcal{E} \rangle$ | Universal AI Intermediate Representation (UAIR) graph |
| $\mathcal{V}$ | Set of all computation nodes: $V_{tensor} \cup V_{quantum} \cup V_{agent}$ |
| $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ | Directed edges representing data/control dependencies |
| $\mathcal{H}$ | Hardware topology graph (GPUs, QPUs, CPUs, interconnects) |
| $\Pi$ | Set of all valid structural permutations (graph rewrites) |
| $\mathcal{S}$ | Runtime state space: $\mathcal{S} = \mathcal{G} \times \mathcal{H} \times \mathcal{M}$ |
| $\mathcal{M}$ | Memory state (working, semantic, procedural, compressed) |

## Scalars and Functions

| Symbol | Definition |
|:---|:---|
| $\mathcal{L}(\mathcal{G}, \mathcal{H})$ | Global multi-objective loss function |
| $\alpha, \beta, \gamma, \delta, \varepsilon, \zeta, \eta, \theta, \iota$ | Scalarization weights for loss components |
| $\pi \in \Pi$ | A single structural permutation (rewrite rule) |
| $w$ | Localized qubit width of an isolated verification subgraph |
| $K$ | Number of distributed agents participating in consensus |
| $N$ | Number of benchmark episodes / repetitions |
| $T$ | Number of evolution timesteps |

## Operators

| Symbol | Definition |
|:---|:---|
| $\pi(\mathcal{G}) \to \mathcal{G}'$ | Application of permutation $\pi$ to graph $\mathcal{G}$ |
| $\mathcal{G} \equiv \mathcal{G}'$ | Semantic equivalence: $\forall x,\ \text{Output}(\mathcal{G}, x) = \text{Output}(\mathcal{G}', x)$ |
| $\text{do}(X := x)$ | Pearl's do-calculus intervention operator |
| $\nabla_\theta \mathcal{L}$ | Gradient of the loss w.r.t. learnable scheduling parameters |

## Complexity Notation

| Symbol | Definition |
|:---|:---|
| $\mathcal{O}(\cdot)$ | Asymptotic upper bound (worst-case) |
| $\Omega(\cdot)$ | Asymptotic lower bound (best-case) |
| $\Theta(\cdot)$ | Tight asymptotic bound (expected-case) |