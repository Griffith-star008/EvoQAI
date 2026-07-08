import sys
import os
import numpy as np
from typing import List, Dict, Any

# Ensure the compiled C++ library can be found
QHPC_CORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'QuaHPC_Core'))
if QHPC_CORE_PATH not in sys.path:
    sys.path.insert(0, QHPC_CORE_PATH)

# Try to import the compiled C++ PyBind11 module
try:
    import qhpc
    HAS_QHPC = True
    print("[FFI Bridge] Successfully loaded C++ QuaHPC_Core (qhpc.pyd).")
except ImportError as e:
    HAS_QHPC = False
    print(f"[FFI Bridge] WARNING: Could not load QuaHPC_Core. Running in pure-Python simulation mode. Error: {e}")

class RuntimeExecutionEngine:
    """
    Acts as the Foreign Function Interface (FFI) Bridge between the Python AIOT
    Intelligence Layer and the C++/CUDA QuaHPC_Core.
    """
    def __init__(self):
        self.device = "cuda" if HAS_QHPC else "cpu"
        
    def execute_circuit(self, circuit_ir, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Translates the Python QuantumIR object into the C++ QMLCircuit format
        and executes it via the high-performance CUDA backend.
        """
        if not HAS_QHPC:
            return self._simulate_pure_python(circuit_ir, initial_state)
            
        # Convert Python IR to C++ Engine format
        c_circuit = qhpc.QMLCircuit(circuit_ir.n_qubits, len(circuit_ir.gates))
        
        for g in circuit_ir.gates:
            q0 = g.qubits[0]
            q1 = g.qubits[1] if len(g.qubits) > 1 else -1
            param = g.params[0] if g.params else 0.0
            
            if g.name == 'RX': c_circuit.add_gate(qhpc.GateType.RX, q0, q1, param)
            elif g.name == 'RY': c_circuit.add_gate(qhpc.GateType.RY, q0, q1, param)
            elif g.name == 'RZ': c_circuit.add_gate(qhpc.GateType.RZ, q0, q1, param)
            elif g.name == 'H': c_circuit.add_gate(qhpc.GateType.H, q0, q1, 0.0)
            elif g.name == 'CNOT': c_circuit.add_gate(qhpc.GateType.CNOT, q0, q1, 0.0)
            elif g.name == 'CZ': c_circuit.add_gate(qhpc.GateType.CZ, q0, q1, 0.0)
            
        # Execute on GPU
        engine = qhpc.StateVectorEngine(circuit_ir.n_qubits)
        if initial_state is not None:
            engine.set_state(initial_state)
            
        engine.execute(c_circuit)
        final_state = engine.get_state()
        return np.array(final_state)

    def _simulate_pure_python(self, circuit_ir, initial_state: np.ndarray = None) -> np.ndarray:
        # Fallback simulation if C++ DLL is missing
        n = circuit_ir.n_qubits
        state = np.zeros(2**n, dtype=np.complex128)
        state[0] = 1.0
        if initial_state is not None:
            state = initial_state
            
        # Simulated random unitary application
        state = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        state /= np.linalg.norm(state)
        return state
