import numpy as np
import torch

class SEAStreamGenerator:
    """
    Generates a streaming dataset with sudden concept drift.
    Based on the standard SEA concepts for data stream mining.
    Features: 3 dimensions (between 0 and 10). Only first 2 are relevant.
    Concept 1: f1 + f2 <= 8
    Concept 2: f1 + f2 <= 9
    """
    def __init__(self, noise_percentage=0.1):
        self.noise = noise_percentage
        self.current_concept = 8.0

    def trigger_drift(self, new_threshold=12.0):
        """Simulate concept drift by changing the classification boundary."""
        self.current_concept = new_threshold

    def get_batch(self, batch_size=32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of streaming data."""
        # Generate 3 features in range [0, 10]
        X = np.random.uniform(0, 10, size=(batch_size, 3))
        
        # Label according to current concept
        y = np.where(X[:, 0] + X[:, 1] <= self.current_concept, -1, 1)
        
        # Add noise
        if self.noise > 0:
            flip_indices = np.random.choice(batch_size, int(batch_size * self.noise), replace=False)
            y[flip_indices] = -y[flip_indices]
            
        # Scale X to [0, pi] for angle embedding in quantum circuit
        X_scaled = X * (np.pi / 10.0)
        
        return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class SineStreamGenerator:
    """
    Sine Stream concept drift (Non-linear).
    Features x1, x2 in [0, 1].
    Label = 1 if x2 < sin(x1 * phase_shift) else 0.
    Drift occurs by changing the phase_shift.
    """
    def __init__(self, noise_percentage=0.05, n_features=4):
        self.noise_percentage = noise_percentage
        self.phase_shift = 1.0
        self.n_features = n_features
        
    def trigger_drift(self, new_phase_shift: float):
        self.phase_shift = new_phase_shift
        
    def get_batch(self, batch_size: int):
        X = np.random.rand(batch_size, 2)
        y = (X[:, 1] < np.sin(X[:, 0] * np.pi * self.phase_shift)).astype(int)
        
        noise_idx = np.random.rand(batch_size) < self.noise_percentage
        y[noise_idx] = 1 - y[noise_idx]
        X_scaled = X * np.pi
        
        # Tile features to match n_features (e.g. [x1, x2] -> [x1, x2, x1, x2])
        # This gives all qubits actual data instead of 0s
        repeats = (self.n_features + 1) // 2
        X_padded = np.tile(X_scaled, (1, repeats))[:, :self.n_features]
        
        y_scaled = y * 2 - 1
        return torch.tensor(X_padded, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)
