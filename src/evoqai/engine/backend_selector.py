import numpy as np

class BackendSelector:
    """
    Context-Aware Backend Selector (Paper 3).
    Uses Multi-Criteria Decision Making (Simple Additive Weighting) to route VQC execution.
    Backends: 'edge_sim', 'cloud_sim', 'qpu_ibmq'
    """
    def __init__(self):
        self.backends = ["edge_sim", "cloud_sim", "qpu_ibmq"]
        
        # Base characteristics for backends [Energy Cost, Latency Cost, Expressivity/Fidelity Gain]
        # Lower cost is better. Higher gain is better.
        self.profiles = {
            "edge_sim":   {"energy": 0.2, "latency": 0.1, "fidelity": 0.5},
            "cloud_sim":  {"energy": 0.8, "latency": 0.5, "fidelity": 0.9}, # High energy due to transmission
            "qpu_ibmq":   {"energy": 0.9, "latency": 1.0, "fidelity": 1.0}  # High latency due to queue
        }

    def select_backend(self, current_circuit_depth: int, data_complexity: float, battery_level: float) -> str:
        """
        Dynamically adjusts weights based on telemetry and evaluates SAW.
        """
        # Dynamic Weights (alpha: energy, beta: latency, gamma: fidelity)
        # If battery is low, prioritize energy.
        alpha = 1.0 - battery_level + 0.1 
        
        # If data is complex or circuit is deep, prioritize fidelity/expressivity.
        gamma = data_complexity + (current_circuit_depth * 0.1)
        
        # Base latency weight
        beta = 0.5 
        
        # Normalize weights
        total_weight = alpha + beta + gamma
        w_e = alpha / total_weight
        w_l = beta / total_weight
        w_f = gamma / total_weight
        
        best_backend = None
        best_score = -float('inf')
        
        for backend in self.backends:
            prof = self.profiles[backend]
            
            # Penalize deep circuits on edge simulator
            if backend == "edge_sim" and current_circuit_depth > 3:
                # Exponential penalty for simulating deep circuits locally
                eff_latency = prof["latency"] * (2 ** (current_circuit_depth - 3))
                eff_energy = prof["energy"] * (2 ** (current_circuit_depth - 3))
            else:
                eff_latency = prof["latency"]
                eff_energy = prof["energy"]
                
            # Score = Gain - Costs
            # We want to maximize the score
            score = (w_f * prof["fidelity"]) - (w_e * eff_energy) - (w_l * eff_latency)
            
            if score > best_score:
                best_score = score
                best_backend = backend
                
        return best_backend
