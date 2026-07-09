from typing import Dict, List, Any

class ReasoningEngine:
    """
    Upgrade 1: Cognitive Layer - Reasoning Engine
    Receives semantic context and operational state, then infers the system's current objectives.
    """
    def __init__(self):
        self.priority_queue = []

    def infer_objectives(self, semantics: List[str], hardware_state: Dict[str, float]) -> List[str]:
        objectives = []
        
        if "OVERHEATING_RISK" in semantics or hardware_state.get("battery", 1.0) < 0.2:
            objectives.append("MINIMIZE_ENERGY")
            
        if "MECHANICAL_WEAR_DETECTED" in semantics:
            objectives.append("MAXIMIZE_ACCURACY") # We need precise predictions for faults
            
        if hardware_state.get("latency_critical", False):
            objectives.append("MINIMIZE_LATENCY")
            
        if not objectives:
            objectives.append("BALANCED_OPERATION")
            
        print(f"[Reasoning Engine] Inferred Objectives: {objectives}")
        return objectives
