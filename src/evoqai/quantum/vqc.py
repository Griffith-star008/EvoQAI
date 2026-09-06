import pennylane as qml
import torch
import torch.nn as nn

class AdaptiveVQC(nn.Module):
    """
    A Variational Quantum Circuit (VQC) in PyTorch.
    Simulates physical quantum noise (Depolarization) based on circuit depth.
    """
    def __init__(self, n_qubits: int, n_layers: int = 1, base_noise_rate: float = 0.05):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.base_noise_rate = base_noise_rate
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Weights for parameterized layers
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="adjoint")

    def _circuit(self, inputs, weights):
        # Data Encoding
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            
        # Parameterized Layers
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
            
        # Physical Noise Simulation: Depth-induced Depolarization
        # The fidelity decays exponentially with depth: f = (1 - p)^depth
        # A depolarizing channel shrinks the expectation value towards 0.
        fidelity = (1.0 - self.base_noise_rate) ** self.n_layers
        outputs = outputs * fidelity
        
        return outputs

    def add_layer(self):
        with torch.no_grad():
            new_layer = torch.randn(1, self.n_qubits, 3)
            new_weights = torch.cat([self.weights.data, new_layer], dim=0)
            self.weights = nn.Parameter(new_weights)
            self.n_layers += 1

    def remove_layer(self):
        if self.n_layers > 1:
            with torch.no_grad():
                new_weights = self.weights.data[:-1, :, :]
                self.weights = nn.Parameter(new_weights)
                self.n_layers -= 1
