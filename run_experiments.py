import sys
import os
import numpy as np

# Ensure imports work across the project
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from intelligence.representation.encoding_search import DynamicRepresentationLearning
from intelligence.evolution_engine.evolution_loop import SelfEvolvingFramework
from intelligence.context_engine.semantic_parser import SemanticContextEngine
from intelligence.feature_space.feature_generator import AdaptiveFeatureGenerator
from memory.knowledge_base import ExperienceMemory
from core.backend_selector.selector import ContextAwareBackendSelector
from core.runtime.ffi_bridge import RuntimeExecutionEngine

def run_phd_simulation():
    print("="*60)
    print(" PhD EXPERIMENT: Self-Evolving Quantum AIOT Framework")
    print("="*60)

    # 1. Initialize Subsystems
    semantic_parser = SemanticContextEngine()
    feature_gen = AdaptiveFeatureGenerator(max_features=8)
    rep_learner = DynamicRepresentationLearning()
    framework = SelfEvolvingFramework()
    backend_selector = ContextAwareBackendSelector()
    memory = ExperienceMemory()
    runtime = RuntimeExecutionEngine()
    
    print(f"\n[INIT] Active Execution Engine: {runtime.device.upper()}")

    # 2. Simulate Incoming IoT Sensor Data & Semantic Context (Directions 5, 6, 1)
    print("\n[PHASE 1] Semantic Context & Representation Learning")
    raw_sensor_dict = {"temperature": 92.5, "vibration": 7.2, "latency_ms": 45}
    semantics = semantic_parser.extract_operational_knowledge(raw_sensor_dict)
    print(f"  -> Raw Data: {raw_sensor_dict}")
    print(f"  -> Semantic Context Extracted: {semantics}")

    raw_sensor_array = np.random.randn(1, 4) # Simulate 4 raw features
    adaptive_features = feature_gen.generate_feature_space(raw_sensor_array)
    encoding = rep_learner.select_optimal_encoding(adaptive_features, {})
    
    print(f"  -> Adaptive Feature Space Generated: {adaptive_features[0]}")
    print(f"  -> Auto-Selected Encoding Strategy: {encoding}")

    # 3. Backend Selection based on Context (Direction 3)
    print("\n[PHASE 2] Context-Aware Backend Selection")
    context = {
        'circuit_complexity': 30,
        'battery_level': 0.4,
        'latency_critical': True,
        'sensor_noise_level': 0.05
    }
    selected_backend = backend_selector.select_optimal_backend(context)
    print(f"  -> Selected Backend: {selected_backend.name} (Type: {selected_backend.qpu_type})")

    # 4. Evolution & Execution Loop (Directions 2, 4, 7, 9)
    print("\n[PHASE 3] Continuous Evolution & Execution")
    print(f"  -> Initial Quantum IR: {framework.current_ir.gates}")
    
    for step in range(1, 6):
        # Simulate environment drift
        accuracy = 0.92 if step < 3 else 0.65 
        
        print(f"\n  --- T={step} ---")
        state_vector = runtime.execute_circuit(framework.current_ir)
        print(f"  Execution Output Size: {len(state_vector)}")
        
        framework.monitor_performance(accuracy=accuracy, latency=0.2)
        
        # Trigger evolution if drift is detected
        if framework.detect_drift():
            # Apply Meta-Learning Feedback to Feature Space
            feature_gen.update_meta_weights({"loss": 1.0 - accuracy})
            
            # Evolve Quantum Circuit
            framework.trigger_evolution()
            framework.performance_history.clear()
            
            # Save the successful adaptation to Lifelong Memory
            memory.store(str(semantics), framework.current_ir, accuracy=0.9)

    print("\n" + "="*60)
    print(" EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_phd_simulation()
