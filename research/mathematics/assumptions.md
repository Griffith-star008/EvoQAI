# Assumptions

The following assumptions underpin the theoretical analysis of the AQIP framework. Each assumption is explicitly stated so that reviewers can evaluate the scope and limitations of our guarantees.

## Assumption 1: DAG Structure
The UAIR graph $\mathcal{G}$ is always a Directed Acyclic Graph (DAG). Cyclic dependencies are rejected at compile time by the frontend parser. This assumption is critical for topological sorting and complexity bounds.

## Assumption 2: Bounded Verification Subgraph
All structural permutations proposed by the Self-Evolving DAG Optimizer operate on **isolated subgraphs** with bounded qubit width $w \leq w_{max}$. In our implementation, $w_{max} = 20$. This prevents the SMT verification step from encountering exponential state-space explosion.

## Assumption 3: Monotonic Loss Improvement
A structural permutation $\pi$ is deployed if and only if $\mathcal{L}(\pi(\mathcal{G}), \mathcal{H}) < \mathcal{L}(\mathcal{G}, \mathcal{H})$. This strict inequality ensures that the evolution sequence is monotonically decreasing.

## Assumption 4: Hardware Stationarity (Local)
During a single evolution cycle (proposal → verification → deployment), the hardware state $\mathcal{H}$ is assumed to be locally stationary. Hardware drifts (e.g., thermal throttling) are detected at the *next* monitoring cycle and trigger a new evolution proposal.

## Assumption 5: Finite Permutation Space
The set of valid structural permutations $\Pi$ is finite for any bounded graph $\mathcal{G}$. Combined with Assumption 3, this guarantees that the evolution process terminates in finitely many steps.

## Assumption 6: Independent Noise Model
Quantum noise in $V_{quantum}$ nodes is modeled as independent depolarizing noise channels. Correlated noise (e.g., crosstalk between adjacent qubits) is not currently modeled and represents a known limitation.

## Assumption 7: Reliable Inter-Node Communication
For distributed execution, we assume that inter-node communication channels are reliable (no message loss) but potentially slow (bounded latency). Byzantine fault tolerance is not assumed.