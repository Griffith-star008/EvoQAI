import numpy as np
from typing import Dict, Any, List

class Backend:
    def __init__(self, name: str, base_latency: float, base_energy: float, availability: float, qpu_type: str):
        self.name = name
        self.base_latency = base_latency
        self.base_energy = base_energy
        self.availability = availability
        self.qpu_type = qpu_type # 'Statevector', 'TensorNetwork', 'QPU', 'CUDA', 'MPS'

class ContextAwareBackendSelector:
    """
    Research Direction 3: Context-Aware Quantum Backend Selection
    Selects the optimal backend based on circuit complexity, energy cost, inference latency,
    sensor noise, hardware availability, and workload distribution.
    """
    def __init__(self):
        # Initial pool of available backends (Edge + Cloud + QPU)
        self.backends = [
            Backend("Local_SV", base_latency=1.0, base_energy=10.0, availability=1.0, qpu_type='Statevector'),
            Backend("Edge_CUDA", base_latency=0.2, base_energy=40.0, availability=0.9, qpu_type='CUDA'),
            Backend("Cloud_TN", base_latency=15.0, base_energy=200.0, availability=0.99, qpu_type='TensorNetwork'),
            Backend("Cloud_QPU", base_latency=150.0, base_energy=500.0, availability=0.4, qpu_type='QPU')
        ]

    def calculate_backend_score(self, backend: Backend, context: Dict[str, Any]) -> float:
        """
        Calculate a dynamic score for a backend given the current AIOT context.
        Lower score is better.
        """
        circuit_complexity = context.get('circuit_complexity', 10) # qubits * depth
        battery_level = context.get('battery_level', 1.0) # 0.0 to 1.0
        latency_critical = context.get('latency_critical', False)
        sensor_noise_level = context.get('sensor_noise_level', 0.1)
        
        # Penalize high energy consumption if battery is low
        energy_penalty = backend.base_energy * (1.0 / (battery_level + 0.01))
        
        # Penalize latency if task is critical (e.g., real-time control)
        latency_penalty = backend.base_latency * (10.0 if latency_critical else 1.0)
        
        # Hard Constraints
        # Local Statevector cannot handle > 25 qubits effectively
        if circuit_complexity > 250 and backend.name == "Local_SV":
            return float('inf')
        
        # QPU might be reserved for highly complex tasks where noise resilience is managed
        if backend.qpu_type == 'QPU' and circuit_complexity < 50:
            latency_penalty *= 5.0 # Unnecessary overhead for small circuits
            
        # Calculate final aggregated score
        alpha, beta, gamma = 0.4, 0.4, 0.2
        score = (alpha * latency_penalty) + (beta * energy_penalty) + (gamma * (1.0 - backend.availability) * 1000)
        
        return score

    def select_optimal_backend(self, context: Dict[str, Any]) -> Backend:
        """
        Evaluates all available backends and selects the one with the lowest cost score.
        """
        best_backend = None
        best_score = float('inf')
        
        for backend in self.backends:
            score = self.calculate_backend_score(backend, context)
            if score < best_score:
                best_score = score
                best_backend = backend
                
        return best_backend

# Example usage
if __name__ == "__main__":
    selector = ContextAwareBackendSelector()
    context = {
        'circuit_complexity': 300,
        'battery_level': 0.8,
        'latency_critical': False,
        'sensor_noise_level': 0.05
    }
    selected = selector.select_optimal_backend(context)
    print(f"Selected Backend: {selected.name} (Type: {selected.qpu_type})")
