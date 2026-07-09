class IdentityEngine:
    """
    Layer: Identity Engine
    Provides the AQIP framework with Computational Self Identity I(t).
    It understands its own hardware profile, current limitations, and available knowledge capacity.
    """
    def __init__(self):
        # I(t): The self-identity state
        self.identity_state = {
            "platform_version": "AQIP v5.0",
            "hardware_profile": "Edge AIOT Node",
            "compute_capacity": "Low",
            "battery_level": 100.0,
            "knowledge_capacity": "High",
            "current_mode": "exploration"
        }

    def update_identity(self, environment_metrics: dict) -> dict:
        """
        Updates self-identity based on environment wear-and-tear or runtime changes.
        """
        print("\n[Identity Engine] Updating Computational Self-Identity I(t)...")
        
        # Simulate battery drain reducing compute capacity
        battery = environment_metrics.get("battery_level", self.identity_state["battery_level"])
        self.identity_state["battery_level"] = battery
        
        if battery < 30.0:
            self.identity_state["compute_capacity"] = "Critical"
            self.identity_state["current_mode"] = "survival"
            print("[Identity Engine] ALERT: Battery critical. Shifting identity to SURVIVAL mode.")
        else:
            self.identity_state["compute_capacity"] = "Optimal"
            self.identity_state["current_mode"] = "exploitation"
            print("[Identity Engine] Identity state normal. Operating in EXPLOITATION mode.")
            
        return self.identity_state
