class ReflectionEngine:
    """
    Layer 4: Reflection Engine
    Evaluates past execution outcomes. If a failure occurred, it performs 
    Root Cause Analysis (RCA) and autonomously mutates or deletes flawed knowledge.
    """
    def __init__(self):
        pass

    def perform_hindsight_analysis(self, strategy: str, outcome: float, context: dict):
        """
        Reflects on the outcome of a strategy.
        Returns an actionable directive to modify the Knowledge Graph.
        """
        print("\n[Reflection Engine] Initiating Hindsight Root Cause Analysis...")
        
        if outcome < 0.5:
            print(f"[Reflection Engine] RCA: Strategy '{strategy}' failed catastrophically in context {context}.")
            print(f"[Reflection Engine] DIRECTIVE: Autonomously purging '{strategy}' from Procedural Memory to prevent future failures.")
            return {"action": "DELETE_KNOWLEDGE", "target": strategy}
        elif outcome > 0.9:
            print(f"[Reflection Engine] RCA: Strategy '{strategy}' exceeded expectations.")
            print(f"[Reflection Engine] DIRECTIVE: Reinforcing '{strategy}' in Knowledge Graph.")
            return {"action": "REINFORCE_KNOWLEDGE", "target": strategy}
        else:
            print(f"[Reflection Engine] RCA: Strategy '{strategy}' performed adequately. No structural changes needed.")
            return {"action": "NONE", "target": strategy}
