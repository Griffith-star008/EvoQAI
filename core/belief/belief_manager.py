class BeliefManager:
    """
    Layer 3: Belief Engine
    Maintains a probabilistic belief state (confidence) for computational strategies using Bayesian updates.
    Models belief using a simplified Beta Distribution (alpha = successes, beta = failures).
    """
    def __init__(self):
        # Initial belief states for different backends. [alpha, beta]
        # alpha: pseudo-count of successes, beta: pseudo-count of failures
        self.beliefs = {
            "Edge_CUDA": [10.0, 1.0], # Highly confident
            "Cloud_MPS": [5.0, 5.0],  # Uncertain
            "Local_SV": [8.0, 2.0]    # Confident
        }

    def update_belief(self, strategy: str, success: bool):
        """
        Performs a Bayesian update on the belief distribution for a given strategy.
        """
        if strategy not in self.beliefs:
            self.beliefs[strategy] = [1.0, 1.0] # Uninformative prior
            
        if success:
            self.beliefs[strategy][0] += 1.0
            print(f"[Belief Engine] Positive Evidence. Bayesian Update: Confidence in {strategy} INCREASED.")
        else:
            self.beliefs[strategy][1] += 1.0
            print(f"[Belief Engine] Negative Evidence. Bayesian Update: Confidence in {strategy} DECAYED.")
            
    def get_confidence(self, strategy: str) -> float:
        """
        Returns the expected value (mean) of the Beta distribution: alpha / (alpha + beta)
        """
        if strategy not in self.beliefs:
            return 0.5
        alpha, beta = self.beliefs[strategy]
        confidence = alpha / (alpha + beta)
        return confidence
        
    def print_belief_state(self):
        print("\n=== AQIP BELIEF STATE ===")
        for strat, params in self.beliefs.items():
            conf = params[0] / (params[0] + params[1])
            print(f" {strat}: {conf*100:.1f}%")
        print("=========================\n")
