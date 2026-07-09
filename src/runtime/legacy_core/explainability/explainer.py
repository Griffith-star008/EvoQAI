from typing import Dict, Any

class ExplainableRuntime:
    """
    Upgrade 12: Explainable Quantum Runtime
    Translates internal numerical decisions (like Backend Selection or Evolution)
    into human-readable justifications.
    """
    def __init__(self):
        pass

    def explain_backend_selection(self, backend_name: str, context: Dict[str, Any], confidence: float):
        explanation = f"\n=== [X-QML] EXPLAINABLE RUNTIME ===\n"
        explanation += f"Backend Selected: {backend_name}\n"
        
        reason = ""
        if "CUDA" in backend_name and context.get('circuit_complexity', 0) > 50:
            reason = "Circuit complexity is high; offloading to GPU for parallel tensor acceleration."
        elif "SV" in backend_name and context.get('latency_critical', False):
            reason = "Latency is critical; executing locally via Statevector to avoid network round-trip overhead."
        elif "QPU" in backend_name:
            reason = "Noise resilience requested; executing on physical quantum hardware."
        else:
            reason = "Optimal balance of energy and latency found for current IoT context."
            
        explanation += f"Reason: {reason}\n"
        explanation += f"Decision Confidence: {confidence*100:.1f}%\n"
        explanation += "==================================="
        print(explanation)
        return explanation
