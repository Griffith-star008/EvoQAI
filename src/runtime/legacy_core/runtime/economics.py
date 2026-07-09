class RuntimeEconomics:
    """
    Layer: Runtime Economics
    AQIP calculates the true operational cost C(t) of execution.
    C = Compute + Memory + Communication + Energy + Time
    """
    def __init__(self):
        pass

    def calculate_cost(self, backend: str, circuit_depth: int) -> float:
        """
        Calculates total runtime economics cost C(t).
        """
        print("\n[Runtime Economics] Calculating operational cost C(t)...")
        
        if backend == "Cloud_MPS":
            compute = 1.0
            memory = 1.0
            communication = 8.0 # Expensive network cost
            energy = 2.0
            time_cost = 5.0
        elif backend == "Edge_CUDA":
            compute = 4.0
            memory = 4.0
            communication = 0.5 # Local execution
            energy = 6.0
            time_cost = 1.0
        else: # Local_SV
            compute = 2.0
            memory = 1.0
            communication = 0.1
            energy = 1.0
            time_cost = 2.0
            
        # Circuit depth acts as a multiplier on compute and time
        depth_multiplier = max(1.0, circuit_depth / 5.0)
        
        total_cost = (compute + time_cost) * depth_multiplier + memory + communication + energy
        
        print(f"[Runtime Economics] Cost for {backend}: Compute={compute}, Comm={communication}, Energy={energy}")
        print(f"[Runtime Economics] Total Cost C(t) = {total_cost:.2f}")
        return total_cost
