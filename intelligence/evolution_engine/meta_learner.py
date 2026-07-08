class MetaLearningEngine:
    """
    Upgrade 6: Meta-Learning Engine
    The framework learns how to learn by dynamically optimizing the evolutionary hyperparameters
    (mutation rate, learning rate, circuit depth) based on the task context.
    """
    def __init__(self):
        self.mutation_rate = 0.1
        self.learning_rate = 0.01

    def optimize_learning_strategy(self, drift_severity: float) -> dict:
        """
        Meta-learning rule: If drift is severe, the system learns faster and mutates more aggressively.
        """
        if drift_severity > 0.6:
            self.mutation_rate = 0.4
            self.learning_rate = 0.05
            print(f"[Meta-Learning Engine] Severe drift ({drift_severity:.2f}) detected. Aggressive exploration activated.")
        elif drift_severity > 0.3:
            self.mutation_rate = 0.2
            self.learning_rate = 0.02
            print(f"[Meta-Learning Engine] Moderate drift ({drift_severity:.2f}). Adaptive tuning activated.")
        else:
            self.mutation_rate = 0.05
            self.learning_rate = 0.005
            print(f"[Meta-Learning Engine] Stable environment. Exploitation strategy activated.")
            
        return {"mutation_rate": self.mutation_rate, "learning_rate": self.learning_rate}
