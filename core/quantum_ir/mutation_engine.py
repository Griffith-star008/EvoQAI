import numpy as np
import random
from typing import List

class QuantumGate:
    def __init__(self, name: str, qubits: List[int], params: List[float] = None):
        self.name = name
        self.qubits = qubits
        self.params = params or []

    def __repr__(self):
        return f"{self.name}({self.qubits}, {self.params})"

class QuantumIR:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[QuantumGate] = []

    def add_gate(self, gate: QuantumGate):
        self.gates.append(gate)

class MutationEngine:
    """
    Research Direction 2: Evolutionary QuantumIR
    Mutates the quantum circuit IR via Add, Remove, Replace, or Swap operations.
    """
    def __init__(self):
        self.gate_pool = ['RX', 'RY', 'RZ', 'H', 'CNOT', 'CZ']

    def mutate(self, ir: QuantumIR, mutation_rate: float = 0.1) -> QuantumIR:
        new_ir = QuantumIR(ir.n_qubits)
        # Deep copy the circuit
        new_ir.gates = [QuantumGate(g.name, list(g.qubits), list(g.params)) for g in ir.gates]

        # 1. Random Gate Removal
        if random.random() < mutation_rate and len(new_ir.gates) > 0:
            idx = random.randint(0, len(new_ir.gates) - 1)
            removed = new_ir.gates.pop(idx)
            print(f"[Mutation] Removed gate {removed.name} at index {idx}")

        # 2. Random Gate Addition
        if random.random() < mutation_rate:
            g_type = random.choice(self.gate_pool)
            if g_type in ['CNOT', 'CZ']:
                q0, q1 = random.sample(range(new_ir.n_qubits), 2)
                new_ir.add_gate(QuantumGate(g_type, [q0, q1]))
            else:
                q0 = random.randint(0, new_ir.n_qubits - 1)
                new_ir.add_gate(QuantumGate(g_type, [q0], [random.uniform(-np.pi, np.pi)]))
            print(f"[Mutation] Added gate {g_type}")

        # 3. Param Perturbation (Continuous Evolution)
        for g in new_ir.gates:
            if g.params and random.random() < mutation_rate:
                g.params[0] += random.gauss(0, 0.1)
                
        return new_ir
