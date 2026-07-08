from typing import Dict, List

class SemanticContextEngine:
    """
    Research Direction 5: Semantic AIOT Intelligence
    Translates raw numerical sensor data (temperature, pressure) into
    operational knowledge and risk contexts.
    """
    def __init__(self):
        # Rule-based knowledge graph (In the future, this is an LLM or Graph Neural Network)
        self.knowledge_rules = {
            "temperature_critical": lambda data: data.get("temperature", 0.0) > 85.0,
            "vibration_erratic": lambda data: data.get("vibration", 0.0) > 5.0,
            "network_unstable": lambda data: data.get("latency_ms", 0) > 500
        }

    def extract_operational_knowledge(self, raw_sensor_data: Dict[str, float]) -> List[str]:
        """
        Converts raw numbers into semantic tokens representing the state of the machine.
        """
        semantics = []
        
        if self.knowledge_rules["temperature_critical"](raw_sensor_data):
            semantics.append("OVERHEATING_RISK")
        if self.knowledge_rules["vibration_erratic"](raw_sensor_data):
            semantics.append("MECHANICAL_WEAR_DETECTED")
        if self.knowledge_rules["network_unstable"](raw_sensor_data):
            semantics.append("EDGE_DISCONNECT_IMMINENT")
            
        if not semantics:
            semantics.append("NORMAL_OPERATION")
            
        return semantics

# Example usage
if __name__ == "__main__":
    engine = SemanticContextEngine()
    
    # Simulate a failing industrial motor
    sensor_readings = {
        "temperature": 92.5,
        "vibration": 7.2,
        "latency_ms": 45
    }
    
    knowledge = engine.extract_operational_knowledge(sensor_readings)
    print("Raw Data:", sensor_readings)
    print("Semantic Knowledge:", knowledge)
