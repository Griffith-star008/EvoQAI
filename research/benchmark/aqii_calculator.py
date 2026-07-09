class AQIICalculator:
    """
    Autonomous Quantum Intelligence Index (AQII) Calculator.
    Computes a composite score (0-100) based on 8 cognitive dimensions.
    """
    @staticmethod
    def calculate_aqii(metrics: dict) -> float:
        """
        metrics dict should contain normalized values (0.0 to 1.0) for:
        runtime, knowledge, belief, reflection, evolution, robustness, explainability, adaptation
        """
        aqii = (
            0.20 * metrics.get("runtime", 0.0) +
            0.15 * metrics.get("knowledge", 0.0) +
            0.15 * metrics.get("belief", 0.0) +
            0.10 * metrics.get("reflection", 0.0) +
            0.10 * metrics.get("evolution", 0.0) +
            0.10 * metrics.get("robustness", 0.0) +
            0.10 * metrics.get("explainability", 0.0) +
            0.10 * metrics.get("adaptation", 0.0)
        )
        
        # Scale to 0-100
        return aqii * 100.0

    @staticmethod
    def get_grade(aqii_score: float) -> str:
        if aqii_score >= 90.0:
            return "Excellent (Autonomous)"
        elif aqii_score >= 75.0:
            return "Good (Adaptive)"
        elif aqii_score >= 50.0:
            return "Fair (Reactive)"
        else:
            return "Poor (Static)"
