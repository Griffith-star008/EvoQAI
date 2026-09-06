class CausalDigitalTwin:
    """
    Structural Causal Model (SCM) Digital Twin.
    Predicts the causal effect of a structural mutation (do-intervention) on the QPU's fidelity.
    """
    def __init__(self, base_noise_rate: float, noise_threshold: float = 0.70):
        self.base_noise_rate = base_noise_rate
        self.noise_threshold = noise_threshold

    def evaluate_mutation(self, current_layers: int, mutation_type: str) -> bool:
        """
        Evaluate if a mutation is SAFE to deploy.
        Returns True if SAFE, False if UNSAFE.
        """
        if mutation_type == "add_layer":
            proposed_layers = current_layers + 1
        elif mutation_type == "remove_layer":
            proposed_layers = current_layers - 1
        else:
            proposed_layers = current_layers
            
        # Causal Equation: Fidelity = (1 - p)^L
        predicted_fidelity = (1.0 - self.base_noise_rate) ** proposed_layers
        
        # If predicted fidelity drops below threshold, the circuit will collapse to white noise.
        is_safe = predicted_fidelity >= self.noise_threshold
        
        if not is_safe:
            print(f"[CausalTwin] 🛑 UNSAFE INTERVENTION DETECTED: Adding layer {proposed_layers} drops fidelity to {predicted_fidelity:.2f} (Threshold: {self.noise_threshold}). Mutation REJECTED.")
            
        return is_safe
