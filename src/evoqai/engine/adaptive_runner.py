import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.evoqai.engine.backend_selector import BackendSelector

class AdaptiveEngine:
    """
    Online training loop with Safe Evolution and Context-Aware Backend Routing.
    """
    def __init__(self, model, causal_twin=None, lr=0.1, window_size=20, is_safe_mode=True):
        self.model = model
        self.causal_twin = causal_twin
        self.is_safe_mode = is_safe_mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.window_size = window_size
        self.accuracy_history = []
        self.drift_detected = False
        
        # Phase 3: Hardware router
        self.backend_selector = BackendSelector()
        self.simulated_battery = 1.0 # Starts at 100%

    def train_step(self, x, y):
        # 1. Hardware Routing (Paper 3)
        # Drain battery over time
        self.simulated_battery -= 0.005 
        data_complexity = 0.9 if self.drift_detected else 0.4
        
        optimal_backend = self.backend_selector.select_backend(
            current_circuit_depth=self.model.n_layers, 
            data_complexity=data_complexity,
            battery_level=self.simulated_battery
        )
        
        # If mapping to Qiskit Aer/IBMQ
        target_qml_backend = "default.qubit"
        if optimal_backend == "qpu_cloud":
            target_qml_backend = "qiskit.ibmq"
        elif optimal_backend == "qpu_local":
            target_qml_backend = "qiskit.aer"
            
        if self.model.backend != target_qml_backend:
            print(f"[BackendSelector] Routing execution to: {optimal_backend} (Battery: {self.simulated_battery:.2f})")
            # In a real run, switching dynamically per batch is expensive, but conceptually valid.
            # self.model.switch_backend(target_qml_backend) # Disabled in simulation for speed
        
        # 2. Forward & Backward Pass
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        
        preds = torch.sign(out)
        acc = (preds == y).float().mean().item()
        
        self.accuracy_history.append(acc)
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)
            
        self._detect_and_adapt()
        
        return loss.item(), acc

    def _detect_and_adapt(self):
        if len(self.accuracy_history) == self.window_size:
            avg_acc = np.mean(self.accuracy_history)
            
            if avg_acc < 0.60 and not self.drift_detected:
                print(f"[AdaptiveEngine {'SAFE' if self.is_safe_mode else 'UNSAFE'}] Drift detected! Avg Acc: {avg_acc:.2f}.")
                
                mutation_safe = True
                if self.is_safe_mode and self.causal_twin is not None:
                    mutation_safe = self.causal_twin.evaluate_mutation(self.model.n_layers, "add_layer")
                    
                if mutation_safe:
                    print(f" -> Deploying Mutation (Adding Layer).")
                    self.model.add_layer()
                    self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
                else:
                    print(f" -> Mutation blocked by Causal Twin. Adapting via Learning Rate decay instead.")
                    for g in self.optimizer.param_groups:
                        g['lr'] = 0.01
                        
                self.accuracy_history = []
                self.drift_detected = True
                
            elif avg_acc > 0.80:
                self.drift_detected = False
