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
