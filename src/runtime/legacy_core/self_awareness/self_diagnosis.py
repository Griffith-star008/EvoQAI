import numpy as np

class ConfidenceEstimator:
    """
    Upgrade 4: Self-Awareness - Confidence Estimator
    Estimates the uncertainty of the quantum prediction. Instead of just outputting
    a value, the system knows how "sure" it is about that value.
    """
    def __init__(self):
        pass

    def estimate_confidence(self, quantum_state_vector: np.ndarray, noise_level: float) -> float:
        """
        Confidence is estimated based on the purity of the state (entropy) 
        and the environmental noise level.
        """
        # Calculate Shannon entropy of the measurement probabilities
        probabilities = np.abs(quantum_state_vector)**2
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Max entropy for N qubits is N. Let's normalize it roughly.
        # Higher entropy = lower confidence. High noise = lower confidence.
        normalized_entropy = min(entropy / np.log2(len(quantum_state_vector)), 1.0)
        
        confidence = (1.0 - normalized_entropy) * (1.0 - noise_level)
        return float(max(0.0, min(1.0, confidence)))

class SelfDiagnosis:
    """
    Upgrade 4: Self-Awareness - Self Diagnosis
    The system reflects on its own operational status (e.g. "I am running slowly" 
    or "My predictions are uncertain").
    """
    def __init__(self):
        self.estimator = ConfidenceEstimator()

    def reflect(self, accuracy: float, latency: float, state_vector: np.ndarray, noise: float) -> str:
        confidence = self.estimator.estimate_confidence(state_vector, noise)
        
        print(f"[Self-Awareness] Reflection: Acc={accuracy:.2f}, Latency={latency:.2f}, Confidence={confidence:.2f}")
        
        if confidence < 0.4:
            return "DIAGNOSIS_UNCERTAIN_PREDICTIONS"
        if latency > 1.0:
            return "DIAGNOSIS_COMPUTE_BOTTLENECK"
        if accuracy < 0.7:
            return "DIAGNOSIS_CONCEPT_DRIFT"
            
        return "DIAGNOSIS_HEALTHY"
