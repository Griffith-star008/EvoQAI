from typing import Dict, Any, List

class ExperienceMemory:
    """
    Research Direction 9: Lifelong Quantum AIOT
    The system remembers previous environments (contexts) and transfers
    optimized QuantumIRs (knowledge) to new, similar environments to avoid
    retraining from scratch.
    """
    def __init__(self):
        # Database mapping semantic environments to optimized quantum circuits
        self.memory_bank: Dict[str, Any] = {}

    def store(self, environment_context: str, optimal_quantum_ir: Any, accuracy: float):
        """
        Stores an evolved circuit into the knowledge base if its performance is high.
        """
        if accuracy > 0.85:
            self.memory_bank[environment_context] = optimal_quantum_ir
            print(f"[Memory] Stored optimized circuit for context: '{environment_context}' (Acc: {accuracy:.2f})")
        else:
            print(f"[Memory] Circuit rejected (Acc: {accuracy:.2f} too low to memorize)")

    def transfer_knowledge(self, new_environment: str) -> Any:
        """
        Searches the knowledge base for a similar past environment to jumpstart
        the evolution process in a new environment.
        """
        # Simplistic transfer mechanism: Check for semantic substring overlap
        # In a real system, this would use semantic vector embeddings and cosine similarity
        for known_env, ir in self.memory_bank.items():
            if known_env in new_environment or new_environment in known_env:
                print(f"[Memory] Transferring knowledge: Found match between '{known_env}' and '{new_environment}'")
                return ir
        
        print(f"[Memory] No prior experience found for '{new_environment}'. Starting evolution from scratch.")
        return None

# Example usage
if __name__ == "__main__":
    memory = ExperienceMemory()
    
    # Store past experience
    memory.store("OVERHEATING_RISK", ["H(0)", "CNOT(0,1)", "RX(1)"], accuracy=0.91)
    memory.store("NORMAL_OPERATION", ["H(0)", "H(1)"], accuracy=0.98)
    
    # Transfer knowledge to a new but similar environment
    new_context = "OVERHEATING_RISK_WITH_NOISE"
    retrieved_ir = memory.transfer_knowledge(new_context)
    print("Retrieved IR:", retrieved_ir)
