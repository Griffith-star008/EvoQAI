import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from core.belief.belief_manager import BeliefManager
from core.reflection.reflection_manager import ReflectionEngine
from knowledge.memory.memory_manager import HierarchicalMemory
from core.world_model.predictive_model import WorldModel
from core.self_theorizing.hypothesis_generator import HypothesisGenerator
from core.explainability.explainer import ExplainableRuntime

class MultiAgentAQIPKernel:
    """
    Layer 13: Multi-Agent AQIP Coordinator (v4.0 Final)
    Coordinates Belief, Reflection, World Model, Knowledge, and Execution.
    """
    def __init__(self):
        print("Booting AQIP v4.0 Multi-Agent Cognitive Kernel...")
        
        self.world_model = WorldModel()
        self.belief_manager = BeliefManager()
        self.reflection_engine = ReflectionEngine()
        self.memory = HierarchicalMemory()
        self.theorizer = HypothesisGenerator()
        self.explainer = ExplainableRuntime()

    def run_cognitive_loop(self, environment_state: dict):
        print("\n" + "="*70)
        print(" [AQIP KERNEL v4.0] INITIATING UNIFIED COGNITIVE LOOP")
        print("="*70)
        
        # 1. Predict Future State
        future_state = self.world_model.predict_future_state(environment_state)
        
        # 2. Check Belief State
        self.belief_manager.print_belief_state()
        
        # Simulate selecting Cloud_MPS which currently has low/uncertain belief
        strategy = "Cloud_MPS"
        confidence = self.belief_manager.get_confidence(strategy)
        print(f"[Kernel] Selected Strategy: {strategy} with Confidence {confidence*100:.1f}%")
        
        # 3. Simulate a Catastrophic Failure on Cloud_MPS due to latency
        print("\n[Runtime] Executing Strategy...")
        simulated_outcome = 0.2 # Failure!
        
        # 4. Update Belief (Bayesian Decay)
        self.belief_manager.update_belief(strategy, success=False)
        self.belief_manager.print_belief_state()
        
        # 5. Reflect & Auto-Mutate Knowledge Graph
        directive = self.reflection_engine.perform_hindsight_analysis(strategy, simulated_outcome, environment_state)
        
        if directive["action"] == "DELETE_KNOWLEDGE":
            # Simulate deleting from procedural memory
            if strategy in self.memory.procedural_memory:
                del self.memory.procedural_memory[strategy]
            print(f"[Kernel] Executed Directive: Purged '{directive['target']}' from active policy.")
            
        # 6. Self-Theorizing
        self.theorizer.generate_hypothesis(failures=5, context="LATENCY")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    kernel = MultiAgentAQIPKernel()
    
    # Force a scenario where Cloud backend fails due to high latency/noise
    env_state = {"temperature": 40.0, "vibration": 1.0, "latency": "CRITICAL"}
    kernel.run_cognitive_loop(env_state)
