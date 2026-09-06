import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class AdaptiveEngine:
    """
    Runs the online training loop.
    Detects accuracy drops (Concept Drift) and triggers structural evolution on the VQC.
    """
    def __init__(self, model, lr=0.01, window_size=50):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.window_size = window_size
        self.accuracy_history = []
        self.drift_detected = False

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy (-1 or 1)
        preds = torch.sign(out)
        acc = (preds == y).float().mean().item()
        
        self.accuracy_history.append(acc)
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)
            
        self._detect_and_adapt()
        
        return loss.item(), acc

    def _detect_and_adapt(self):
        """
        Simple drift detection: If average accuracy over the window drops below 0.6,
        we trigger a mutation (add a layer to increase capacity).
        """
        if len(self.accuracy_history) == self.window_size:
            avg_acc = np.mean(self.accuracy_history)
            if avg_acc < 0.60 and not self.drift_detected:
                print(f"[AdaptiveEngine] Drift detected! Avg Acc: {avg_acc:.2f}. Mutating Circuit (Adding Layer)...")
                self.model.add_layer()
                # Reset optimizer state for new parameters
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
                # Flush history to allow recovery
                self.accuracy_history = []
                self.drift_detected = True
            elif avg_acc > 0.80:
                self.drift_detected = False # Reset flag when stable
