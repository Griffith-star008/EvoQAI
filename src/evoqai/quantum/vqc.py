import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class AdaptiveVQC(nn.Module):
    """
    A Variational Quantum Circuit (VQC) in PyTorch.
    Supports dynamic addition/removal of layers (Evolutionary Adaptation).
    """
    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # We start with weights for `n_layers`
        # Shape: (layers, qubits, 3) for RZ, RY, RZ rotations
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
        # Create the qnode
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method="adjoint")

    def _circuit(self, inputs, weights):
        # Data Encoding (Angle Embedding)
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
            
        # Parameterized Layers
        for layer_idx in range(weights.shape[0]):
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Rotations
            for i in range(self.n_qubits):
                qml.RZ(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
                qml.RZ(weights[layer_idx, i, 2], wires=i)
                
        # Measurement (Expectation value of PauliZ on first qubit)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        # We process a batch of inputs
        # QNode expects 1D arrays for inputs if not broadcasted, but pennylane supports batching.
        # We will iterate for simplicity or rely on pennylane broadcasting if configured.
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            outputs[i] = self.qnode(x[i], self.weights)
        return outputs

    def add_layer(self):
        """Evolutionary operator: Add a parameterized layer to increase capacity."""
        with torch.no_grad():
            new_layer = torch.randn(1, self.n_qubits, 3)
            # Concatenate old weights with new layer
            new_weights = torch.cat([self.weights.data, new_layer], dim=0)
            self.weights = nn.Parameter(new_weights)
            self.n_layers += 1

    def remove_layer(self):
        """Evolutionary operator: Remove a layer to reduce noise/overfitting."""
        if self.n_layers > 1:
            with torch.no_grad():
                new_weights = self.weights.data[:-1, :, :]
                self.weights = nn.Parameter(new_weights)
                self.n_layers -= 1
