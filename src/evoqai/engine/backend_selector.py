class BackendSelector:
    """
    Context-Aware Quantum Backend Selection for Resource-Constrained AIoT.
    Multi-objective optimization router.
    """
    def __init__(self, w_latency=0.4, w_energy=0.3, w_fidelity=0.3):
        self.w_latency = w_latency
        self.w_energy = w_energy
        self.w_fidelity = w_fidelity

    def select_backend(self, current_circuit_depth: int, data_complexity: float, battery_level: float) -> str:
        """
        Evaluate context and return the optimal backend:
        'cpu' -> classical fallback / default.qubit
        'qpu_local' -> qiskit.aer simulated QPU for low latency edge
        'qpu_cloud' -> qiskit.ibmq for high fidelity complex circuits
        """
        # Heuristic rules simulating a Pareto front optimization
        
        # 1. Critical Battery -> Classical CPU
        if battery_level < 0.2:
            return "cpu"
            
        # 2. High Complexity + Good Battery -> Cloud QPU
        if data_complexity > 0.8 and battery_level > 0.5:
            # Cloud has high queue latency but high expressivity
            score_cloud = self.w_fidelity * 0.9 - self.w_latency * 0.8
            score_edge = self.w_fidelity * 0.6 - self.w_latency * 0.2
            
            if score_cloud > score_edge:
                return "qpu_cloud"
                
        # 3. Default -> Local Simulated QPU (Edge)
        return "qpu_local"
