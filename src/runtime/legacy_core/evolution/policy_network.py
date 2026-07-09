class EvolutionPolicyNetwork:
    """
    Layer 8: Evolution Policy Network
    A neural-network-inspired policy that decides the mutation strategy 
    (when to evolve, what to evolve) based on state, rather than pure randomness.
    """
    def __init__(self):
        pass

    def determine_mutation_strategy(self, accuracy_drop: float, noise_level: float) -> str:
        """
        Outputs a specific mutation action space based on conditions.
        """
        print("\n[Evolution Policy] Analyzing degradation state to determine mutation strategy...")
        
        if accuracy_drop > 0.2 and noise_level > 0.5:
            strategy = "MUTATION_STRUCTURAL_PRUNING" # High noise -> cut gates
        elif accuracy_drop > 0.2:
            strategy = "MUTATION_EXPAND_DEPTH" # Low noise, high drop -> learn more complex features
        else:
            strategy = "MUTATION_PARAMETER_SHIFT" # Minor drop -> just tune weights
            
        print(f"[Evolution Policy] Policy Network Selected Strategy: {strategy}")
        return strategy
