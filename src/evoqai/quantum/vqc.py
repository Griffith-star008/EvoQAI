import pennylane as qml
import torch
import torch.nn as nn
import os
from dotenv import load_dotenv

# Load env variables for IBMQ
load_dotenv('credentials.env')

class AdaptiveVQC(nn.Module):
    """
    A Variational Quantum Circuit (VQC) in PyTorch.
    Supports dynamic backend switching (CPU, Aer, IBMQ Cloud).
    """
    def __init__(self, n_qubits: int, n_layers: int = 1, base_noise_rate: float = 0.05, backend: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.base_noise_rate = base_noise_rate
        self.backend = backend
        
        self.device = self._initialize_device(backend, n_qubits)
        
        # Weights for parameterized layers
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="adjoint" if backend=="default.qubit" else "parameter-shift")

    def _initialize_device(self, backend: str, wires: int):
        print(f"[Hardware] Initializing QPU Backend: {backend}")
        if backend == "qiskit.ibmq":
            token = os.getenv("IBM_QUANTUM_TOKEN")
            if token and token != "your_ibm_quantum_token_here":
                try:
                    from qiskit_ibm_provider import IBMProvider
                    IBMProvider.save_account(token, overwrite=True)
                    provider = IBMProvider()
                    return qml.device('qiskit.ibmq', wires=wires, backend='ibm_kyiv', provider=provider)
                except Exception as e:
                    print(f"Failed to connect to IBMQ: {e}. Falling back to qiskit.aer")
                    return qml.device('qiskit.aer', wires=wires)
            else:
                print("No IBM Token found in credentials.env. Falling back to qiskit.aer")
                return qml.device('qiskit.aer', wires=wires)
                
        elif backend == "qiskit.aer":
            return qml.device('qiskit.aer', wires=wires)
        else:
            return qml.device("default.qubit", wires=wires)

    def switch_backend(self, new_backend: str):
        if self.backend != new_backend:
            self.backend = new_backend
            self.device = self._initialize_device(new_backend, self.n_qubits)
            self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="adjoint" if new_backend=="default.qubit" else "parameter-shift")

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            
        for layer_idx in range(weights.shape[0]):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            for i in range(self.n_qubits):
                qml.RZ(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
                
        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            outputs[i] = self.qnode(x[i], self.weights)
            
        fidelity = (1.0 - self.base_noise_rate) ** self.n_layers
        outputs = outputs * fidelity
        
        return outputs

    def add_layer(self):
        """Standard Phase 1 mutation: Add parameterized entanglement layer."""
        new_weights = nn.Parameter(torch.randn(1, self.n_qubits, 3) * 0.1)
        self.weights = nn.Parameter(torch.cat([self.weights, new_weights], dim=0))
        self.n_layers += 1
        
    def reinitialize_layer(self, layer_idx=None):
        """Structural reset mutation: Reset a layer to jump out of barren plateaus."""
        if layer_idx is None:
            layer_idx = self.n_layers - 1 # reset last layer
        with torch.no_grad():
            self.weights[layer_idx].normal_(mean=0.0, std=0.5)
            
    def mutate(self, mutation_type: str = "add_layer"):
        """Dynamic architectural search mutation operator."""
        if mutation_type == "add_layer":
            self.add_layer()
        elif mutation_type == "reinit_layer":
            self.reinitialize_layer()
        else:
            raise ValueError(f"Unknown mutation type {mutation_type}")

    def remove_layer(self):
        if self.n_layers > 1:
            with torch.no_grad():
                new_weights = self.weights.data[:-1, :, :]
                self.weights = nn.Parameter(new_weights)
                self.n_layers -= 1
