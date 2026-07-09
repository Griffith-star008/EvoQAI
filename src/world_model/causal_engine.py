class CausalEngine:
    """
    World Model: Causal Reasoner and Do-Calculus Engine.
    Allows the framework to ask counterfactuals ("What if?") before executing policies.
    """
    def __init__(self):
        # A simple causal graph representing dependencies: policy -> latency -> utility
        self.causal_graph = {
            "scheduling_policy": ["latency", "energy"],
            "compiler_fusion": ["latency", "memory", "quantum_noise"],
            "latency": ["utility"],
            "energy": ["utility"],
            "memory": ["utility"],
            "quantum_noise": ["utility"]
        }
        
    def do_intervention(self, intervention: dict, context: dict) -> dict:
        """
        Simulates the causal effect of an intervention using Pearl's do-calculus concept.
        e.g., intervention = {"scheduling_policy": "dynamic_stealing"}
        """
        print(f"\n[Causal Engine] Simulating Intervention: do({intervention})")
        
        simulated_context = context.copy()
        
        # Apply hypothetical effects
        if intervention.get("scheduling_policy") == "dynamic_stealing":
            simulated_context["latency"] = max(1.0, context.get("latency", 10.0) * 0.7)
            simulated_context["energy"] = context.get("energy", 50.0) * 1.2 # Costs more energy
            
        elif intervention.get("compiler_fusion") == "aggressive":
            simulated_context["memory"] = context.get("memory", 100.0) * 0.8
            simulated_context["quantum_noise"] = context.get("quantum_noise", 0.1) * 0.9
            
        # Recompute downstream utility based on causal graph
        utility = (
            (100.0 / simulated_context.get("latency", 1.0)) + 
            (100.0 / simulated_context.get("energy", 1.0))
        )
        simulated_context["projected_utility"] = utility
        
        print(f"[Causal Engine] Counterfactual Outcome -> Projected Utility: {utility:.2f}")
        return simulated_context

    def evaluate_counterfactual(self, context: dict):
        """
        The Scientist Agent uses this to evaluate multiple hypotheses before proposing one.
        """
        print("[Causal Engine] Evaluating Counterfactual Hypotheses...")
        opt_1 = self.do_intervention({"scheduling_policy": "dynamic_stealing"}, context)
        opt_2 = self.do_intervention({"compiler_fusion": "aggressive"}, context)
        
        if opt_1["projected_utility"] > opt_2["projected_utility"]:
            return "dynamic_stealing"
        return "aggressive_fusion"
