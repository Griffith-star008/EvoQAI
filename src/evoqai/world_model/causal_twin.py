import numpy as np
import json
import os
import datetime

class CausalDigitalTwin:
    """
    Structural Causal Model (SCM) Digital Twin for NISQ Hardware.
    Models the causal effect of interventions (do-calculus) considering physical noise bounds:
    - T1 (Thermal Relaxation)
    - T2 (Dephasing)
    - Gate errors (Depolarizing)
    """
    def __init__(self, t1_time=100.0, t2_time=100.0, gate_time=0.1, gate_error_rate=0.01, noise_threshold=0.55):
        # Physical Hardware Parameters (microseconds)
        self.t1_time = t1_time
        self.t2_time = t2_time
        self.gate_time = gate_time
        self.gate_error_rate = gate_error_rate
        self.noise_threshold = noise_threshold
        
        # Logging for interventions
        self.log_file = "experiments/reports/intervention_logs.jsonl"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _predict_fidelity(self, current_layers: int, proposed_layers: int, n_qubits: int) -> float:
        """
        SCM Equation: 
        F = F_depolarizing * F_relaxation * F_dephasing
        """
        # Gates per layer: 1 RY + 3 rotations per qubit, plus entanglement
        # Simplified to ~ 5 gates per qubit per layer
        total_gates = proposed_layers * 5 * n_qubits
        total_time = total_gates * self.gate_time
        
        # 1. Depolarizing Channel Effect
        f_depol = (1.0 - self.gate_error_rate) ** total_gates
        
        # 2. Thermal Relaxation Effect (T1)
        f_t1 = np.exp(-total_time / self.t1_time)
        
        # 3. Dephasing Effect (T2)
        f_t2 = np.exp(-total_time / self.t2_time)
        
        # Confounding factors (e.g., cross-talk exponentially increasing with depth)
        cross_talk = 0.99 ** (proposed_layers ** 1.5)
        
        return f_depol * f_t1 * f_t2 * cross_talk

    def evaluate_mutation(self, current_layers: int, n_qubits: int, mutation_type: str) -> bool:
        """
        Evaluate if a mutation is SAFE to deploy via do-calculus.
        """
        if mutation_type == "add_layer":
            proposed_layers = current_layers + 1
        elif mutation_type == "remove_layer":
            proposed_layers = current_layers - 1
        else:
            proposed_layers = current_layers
            
        predicted_fidelity = self._predict_fidelity(current_layers, proposed_layers, n_qubits)
        is_safe = predicted_fidelity >= self.noise_threshold
        
        # Log Intervention
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "intervention": f"do(L={proposed_layers})",
            "current_L": current_layers,
            "predicted_fidelity": predicted_fidelity,
            "threshold": self.noise_threshold,
            "decision": "SAFE" if is_safe else "UNSAFE",
            "reason": "Fidelity above threshold" if is_safe else "Fidelity collapse predicted due to T1/T2/Depolarization bounds."
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        if not is_safe:
            print(f"[CausalTwin SCM] 🛑 UNSAFE INTERVENTION: do(L={proposed_layers}) yields F={predicted_fidelity:.2f} (T1/T2 constraints). Blocked.")
            
        return is_safe
