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
    def __init__(self, n_qubits=3, n_layers=1, base_noise_rate=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.base_noise_rate = base_noise_rate
        self.backend = "default.qubit"
        self.device = self._initialize_device(self.backend, self.n_qubits)
        
        # Initialize weights for current depth
        self.weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 3))
        
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="adjoint")

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
            try:
                import qiskit
                return qml.device('qiskit.aer', wires=wires)
            except ImportError:
                print("qiskit-aer not installed. Falling back to default.qubit")
                return qml.device("default.qubit", wires=wires)
        else:
            return qml.device("default.qubit", wires=wires)

    def switch_backend(self, new_backend: str):
        if self.backend != new_backend:
            self.backend = new_backend
            self.device = self._initialize_device(new_backend, self.n_qubits)
            diff_method = "adjoint" if new_backend == "default.qubit" else "parameter-shift"
            self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method=diff_method)

    def _circuit(self, inputs, weights):
        # Data Re-uploading + Strong Entanglement
        for layer_idx in range(weights.shape[0]):
            # 1. Data Encoding (Re-uploaded every layer)
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
                
            # 2. Entanglement Ring
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # 3. Parameterized Rotations
            for i in range(self.n_qubits):
                qml.RZ(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
                
        # Return expectation values of all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        # Vectorized batch execution
        outputs_list = torch.stack([torch.stack(self.qnode(x_i, self.weights)) for x_i in x])
        
        # Aggregate across qubits (mean) to form final prediction
        outputs = torch.mean(outputs_list, dim=-1)
        
        outputs = outputs.to(torch.float32)
        fidelity = (1.0 - self.base_noise_rate) ** self.n_layers
        return outputs * fidelity

    def add_layer(self):
        """Standard Phase 1 mutation: Add parameterized entanglement layer.
        Warm-start with identity (zeros) so that the mutation does not destroy prior learned patterns.
        """
        new_weights = nn.Parameter(torch.zeros(1, self.n_qubits, 3))
        self.weights = nn.Parameter(torch.cat([self.weights, new_weights], dim=0))
        self.n_layers += 1
        # Recreate QNode to prevent adjoint differentiation caching bugs when shape changes
        diff_method = "adjoint" if self.backend == "default.qubit" else "parameter-shift"
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method=diff_method)
        
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
