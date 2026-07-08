import sys
import os
import random
from typing import Dict

# Add parent directory to path so we can import the mutation engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.quantum_ir.mutation_engine import QuantumIR, MutationEngine, QuantumGate

class SelfEvolvingFramework:
    """
    Research Direction 4 & 7: Autonomous Quantum Operating Intelligence
    Monitors data drift, triggers evolution, and updates the active QuantumIR.
    """
    def __init__(self):
        self.current_ir = QuantumIR(n_qubits=4)
        self.mutation_engine = MutationEngine()
        self.performance_history = []

        # Baseline circuit
        self.current_ir.add_gate(QuantumGate('H', [0]))
        self.current_ir.add_gate(QuantumGate('CNOT', [0, 1]))

    def monitor_performance(self, accuracy: float, latency: float):
        self.performance_history.append({'acc': accuracy, 'lat': latency})
        print(f"[Monitor] Acc: {accuracy:.2f} | Latency: {latency:.2f}")
        
    def detect_drift(self) -> bool:
        """Detects if the AI model performance has degraded due to environment changes."""
        if len(self.performance_history) < 5:
            return False
            
        recent_acc = sum(x['acc'] for x in self.performance_history[-3:]) / 3.0
        old_acc = sum(x['acc'] for x in self.performance_history[:-3]) / len(self.performance_history[:-3])
        
        # Trigger if accuracy drops by more than 10%
        return recent_acc < old_acc * 0.9

    def trigger_evolution(self):
        print("\n[Evolution Engine] Concept drift detected! Triggering Architecture Search...")
        best_candidate = None
        best_score = float('-inf')

        # Evolutionary Strategy: (1 + lambda)
        lambda_candidates = 5
        for i in range(lambda_candidates):
            print(f"--- Evaluating Candidate {i} ---")
            # High mutation rate to escape local minima
            candidate_ir = self.mutation_engine.mutate(self.current_ir, mutation_rate=0.4)
            
            # Simulate validation score 
            # (In reality, we train/validate this IR on the recent experience buffer)
            simulated_val_score = 0.8 + 0.15 * random.random() - 0.02 * len(candidate_ir.gates)
            
            print(f"  Candidate {i} Score: {simulated_val_score:.3f} | Depth: {len(candidate_ir.gates)}")
            
            if simulated_val_score > best_score:
                best_score = simulated_val_score
                best_candidate = candidate_ir

        print(f"\n[Evolution Engine] Search complete. Updating active QuantumIR (Score: {best_score:.3f})")
        self.current_ir = best_candidate

# Example usage
if __name__ == "__main__":
    framework = SelfEvolvingFramework()
    print("Initial Circuit:", framework.current_ir.gates)
    print("-" * 50)
    
    # Simulate a deployed lifecycle
    for step in range(8):
        print(f"Deployment Step {step}:")
        # Simulate environment drift at step 4 causing accuracy drop
        acc = 0.95 if step < 4 else 0.70 
        framework.monitor_performance(accuracy=acc, latency=0.5)
        
        if framework.detect_drift():
            framework.trigger_evolution()
            framework.performance_history.clear() # Reset history after evolving the model
            print("-" * 50)
            
    print("Final Evolved Circuit:", framework.current_ir.gates)
