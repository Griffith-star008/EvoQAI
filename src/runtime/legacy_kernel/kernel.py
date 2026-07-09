import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from core.state.runtime_state import RuntimeState
from core.contracts.constitution import RuntimeConstitution, ConstitutionViolation
from research.experiment.experiment_engine import ExperimentEngine

class FormalStateKernel:
    """
    Layer: Formal Computational Model (v6.0 Theory)
    The Kernel is no longer a collection of modules. It is a strict Deterministic State Machine.
    """
    def __init__(self):
        print("Booting AQIP v6.0 Formal Computational Theory Kernel...")
        self.experiment_engine = ExperimentEngine()
        self.state = RuntimeState(state_id="S0")

    def transition_observe(self, state: RuntimeState) -> RuntimeState:
        print("[State Transition] -> OBSERVE")
        state.resources["battery"] -= 1.0 # Cost of observing
        return state

    def transition_evolve(self, state: RuntimeState) -> RuntimeState:
        print("[State Transition] -> EVOLVE POLICY")
        # Simulating an evolution attempt
        state.policy = "Quantum_Fourier_Reuploading_v2"
        return state
        
    def transition_unsafe_action(self, state: RuntimeState) -> RuntimeState:
         print("[State Transition] -> CORRUPT KNOWLEDGE (Simulated Attack)")
         state.knowledge.consistency_score = 0.5 # Deliberately break consistency
         return state

    def run_deterministic_cycle(self):
        """
        Executes the state machine loop, enforcing formal theory.
        """
        print("\n" + "="*80)
        print(" [AQIP KERNEL v6.0] INITIATING DETERMINISTIC STATE MACHINE")
        print("="*80)
        
        # Snapshot state before transition
        pre_state = self.state.snapshot()
        
        try:
            # Transition 1: Valid
            self.state = self.transition_observe(self.state)
            RuntimeConstitution.enforce_invariants(pre_state, self.state, "Observe")
            self.experiment_engine.record_experiment(pre_state, self.state, "Observe")
            
            # Transition 2: Invalid (Evolve without evidence history)
            pre_state = self.state.snapshot()
            self.state = self.transition_evolve(self.state)
            # This will raise a ConstitutionViolation because execution_history is empty
            RuntimeConstitution.enforce_invariants(pre_state, self.state, "Evolve_Policy")
            
        except ConstitutionViolation as e:
            print(f"\n[Kernel Exception] Constitution Blocked Transition: {e}")
            print("[Kernel Rollback] Reverting RuntimeState to previous stable snapshot (S0).")
            self.state = pre_state # ROLLBACK
            
        print("\n[Kernel] Final State Integrity Verified.")
        print(f"[Kernel] Current Policy: {self.state.policy}")
        print("="*80 + "\n")

if __name__ == "__main__":
    kernel = FormalStateKernel()
    kernel.run_deterministic_cycle()
