import numpy as np

class BackendSelector:
    """
    Context-Aware Backend Selector (Paper 3).
    Uses Multi-Criteria Decision Making (Simple Additive Weighting) to route VQC execution.
    Backends: 'edge_sim', 'cloud_sim', 'qpu_ibmq'
    """
    def __init__(self):
        self.backends = ["edge_sim", "cloud_sim", "qpu_ibmq"]
        
        # Physical characteristics for backends
        # Energy in Joules per batch, Latency in milliseconds, Fidelity/Expressivity Gain
        # Simulated Edge: Raspberry Pi 4 (5W)
        # Simulated Radio: 5G TX (2W but fast) or WiFi (1W)
        self.profiles = {
            "edge_sim":   {"energy_J": 0.5,  "latency_ms": 150.0, "fidelity": 0.70},
            "cloud_sim":  {"energy_J": 0.2,  "latency_ms": 300.0, "fidelity": 0.95}, # Low local energy (just TX/RX), high latency
            "qpu_ibmq":   {"energy_J": 0.2,  "latency_ms": 5000.0,"fidelity": 1.0}   # Huge queue latency
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
        
        # Max values for normalization
        max_energy = max([p["energy_J"] for p in self.profiles.values()]) * (2 ** 3) # account for depth penalty
        max_latency = max([p["latency_ms"] for p in self.profiles.values()])
        
        for backend in self.backends:
            prof = self.profiles[backend]
            
            # Penalize deep circuits on edge simulator
            if backend == "edge_sim" and current_circuit_depth > 3:
                # Exponential penalty for simulating deep circuits locally
                eff_latency = prof["latency_ms"] * (2 ** (current_circuit_depth - 3))
                eff_energy = prof["energy_J"] * (2 ** (current_circuit_depth - 3))
            else:
                eff_latency = prof["latency_ms"]
                eff_energy = prof["energy_J"]
                
            # Normalize costs to [0, 1]
            norm_latency = eff_latency / max_latency
            norm_energy = eff_energy / max_energy
                
            # Score = Gain - Costs
            # We want to maximize the score
            score = (w_f * prof["fidelity"]) - (w_e * norm_energy) - (w_l * norm_latency)
            
            if score > best_score:
                best_score = score
                best_backend = backend
                
        return best_backend
