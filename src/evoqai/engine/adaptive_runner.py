import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.evoqai.engine.backend_selector import BackendSelector
from src.evoqai.engine.drift_detector import ADWINDriftDetector

class AdaptiveEngine:
    """
    Online training loop with Safe Evolution and Context-Aware Backend Routing.
    Uses ADWIN for statistically robust concept drift detection.
    """
    def __init__(self, model, causal_twin=None, lr=0.1, is_safe_mode=True):
        self.model = model
        self.causal_twin = causal_twin
        self.is_safe_mode = is_safe_mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Phase 4 Upgrade: Real ADWIN detector
        self.drift_detector = ADWINDriftDetector(delta=0.1)
        
        self.backend_selector = BackendSelector()
        self.simulated_battery = 1.0 
        
        # Track accuracy for reporting
        self.accuracy_history = []
        self.recent_acc = 1.0

    def train_step(self, x, y):
        # 1. Hardware Routing (Paper 3)
        self.simulated_battery -= 0.005 
        data_complexity = 0.9 if len(self.drift_detector.window) < 5 else 0.4
        
        optimal_backend = self.backend_selector.select_backend(
            current_circuit_depth=self.model.n_layers, 
            data_complexity=data_complexity,
            battery_level=self.simulated_battery
        )
        
        target_qml_backend = "default.qubit"
        if optimal_backend == "qpu_ibmq":
            target_qml_backend = "qiskit.ibmq"
        elif optimal_backend == "cloud_sim":
            target_qml_backend = "qiskit.aer"
            
        if hasattr(self.model, 'backend') and self.model.backend != target_qml_backend:
            print(f"[AdaptiveEngine] Routing execution to: {optimal_backend} (Battery: {self.simulated_battery:.2f}, Depth: {self.model.n_layers})")
            self.model.switch_backend(target_qml_backend)
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        
        preds = torch.sign(out)
        acc = (preds == y).float().mean().item()
        self.accuracy_history.append(acc)
        
        # For ADWIN, we pass error rate
        error_rate = 1.0 - acc
        self.recent_acc = (self.recent_acc * 0.9) + (acc * 0.1) # EMA
        
        # 3. Detect Drift & Adapt
        drift_found = self.drift_detector.add_element(error_rate)
        
        if drift_found:
            self._adapt()
            
        return loss.item(), acc

    def _adapt(self):
        print(f"[AdaptiveEngine {'SAFE' if self.is_safe_mode else 'UNSAFE'}] ADWIN Drift detected! Recent Acc: {self.recent_acc:.2f}.")
        
        mutation_safe = True
        if self.is_safe_mode and self.causal_twin is not None:
            # Need to pass n_qubits
            n_qubits = getattr(self.model, 'n_qubits', 3)
            mutation_safe = self.causal_twin.evaluate_mutation(self.model.n_layers, n_qubits, "add_layer")
            
        if mutation_safe:
            print(f" -> Deploying Mutation (add_layer).")
            if hasattr(self.model, 'mutate'):
                self.model.mutate("add_layer")
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        else:
            print(f" -> add_layer blocked by Causal Twin SCM. Deploying fallback Mutation (reinit_layer).")
            if hasattr(self.model, 'mutate'):
                self.model.mutate("reinit_layer")
            for g in self.optimizer.param_groups:
                g['lr'] = 0.05
                
        self.drift_detector.reset()
