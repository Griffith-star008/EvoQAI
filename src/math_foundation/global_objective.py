class GlobalObjectiveFunction:
    """
    Mathematical Foundation of the Global Production Framework.
    Every subsystem optimizes this unified loss function.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0, zeta=1.0, eta=1.0, theta=1.0):
        self.weights = {
            "task": alpha,
            "latency": beta,
            "memory": gamma,
            "energy": delta,
            "accuracy": epsilon,
            "quantum": zeta,
            "security": eta,
            "reliability": theta
        }

    def compute_loss(self, metrics: dict) -> float:
        """
        Computes L = a*L_task + b*L_lat + c*L_mem + d*L_eng + e*L_acc + f*L_qnt + g*L_sec + h*L_rel
        """
        loss = 0.0
        for key, weight in self.weights.items():
            # If a metric is missing, default to a high penalty of 1.0
            loss += weight * metrics.get(f"L_{key}", 1.0)
            
        print(f"[Math Foundation] Computed Global Objective Loss L = {loss:.4f}")
        return loss

    def optimize_subsystem(self, subsystem_name: str, current_metrics: dict) -> dict:
        """
        Simulates the global optimizer adjusting a subsystem to minimize L.
        """
        print(f"[Math Foundation] Optimizing Subsystem: {subsystem_name}")
        current_loss = self.compute_loss(current_metrics)
        
        # Simulate gradient descent step reducing latency and energy loss
        optimized_metrics = current_metrics.copy()
        if "L_latency" in optimized_metrics:
            optimized_metrics["L_latency"] *= 0.9
        if "L_energy" in optimized_metrics:
            optimized_metrics["L_energy"] *= 0.85
            
        new_loss = self.compute_loss(optimized_metrics)
        print(f"[Math Foundation] Convergence: Loss reduced from {current_loss:.4f} -> {new_loss:.4f}")
        
        return optimized_metrics
