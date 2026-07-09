class GoalManager:
    """
    Layer: Goal Manager & Utility Function
    Transitions AQIP from naive optimization to Multi-Objective Goal Execution G(t).
    Calculates Intelligence Utility U(t) based on accuracy, adaptability, and energy.
    """
    def __init__(self):
        # Dynamic weights for the utility function [alpha, beta, gamma, delta, epsilon, zeta]
        self.weights = {
            "accuracy": 0.4,
            "adaptability": 0.2,
            "knowledge": 0.1,
            "robustness": 0.1,
            "explainability": 0.05,
            "energy": 0.15
        }

    def set_goal_from_identity(self, identity_state: dict):
        """
        Dynamically adjusts utility weights based on computational self-identity.
        """
        print("\n[Goal Manager] Adjusting Goal G(t) based on Identity I(t)...")
        if identity_state.get("current_mode") == "survival":
            # Prioritize energy and robustness over raw accuracy
            self.weights["accuracy"] = 0.2
            self.weights["energy"] = 0.5
            self.weights["robustness"] = 0.3
            print("[Goal Manager] Goal Shifted: Prioritizing Energy Conservation and Survival.")
        else:
            # Standard intelligence maximization
            self.weights["accuracy"] = 0.5
            self.weights["energy"] = 0.1
            print("[Goal Manager] Goal Shifted: Prioritizing Maximum Accuracy.")

    def calculate_utility(self, metrics: dict) -> float:
        """
        Calculates the Mathematical Intelligence Utility U(t).
        """
        u = (
            self.weights["accuracy"] * metrics.get("accuracy", 0.0) +
            self.weights["adaptability"] * metrics.get("adaptability", 0.0) +
            self.weights["knowledge"] * metrics.get("knowledge_growth", 0.0) +
            self.weights["robustness"] * metrics.get("robustness", 0.0) +
            self.weights["explainability"] * metrics.get("explainability", 0.0) +
            self.weights["energy"] * metrics.get("energy_efficiency", 0.0)
        )
        print(f"[Goal Manager] Calculated Utility Function U(t) = {u:.4f}")
        return u
