import sys
import os
import numpy as np

# Ensure imports work across the project
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from intelligence.representation.encoding_search import DynamicRepresentationLearning
from intelligence.evolution_engine.evolution_loop import SelfEvolvingFramework
from core.backend_selector.selector import ContextAwareBackendSelector
from core.runtime.ffi_bridge import RuntimeExecutionEngine

def run_phd_simulation():
    print("="*60)
    print(" PhD EXPERIMENT: Self-Evolving Quantum AIOT Framework")
    print("="*60)

    # 1. Initialize Subsystems
    rep_learner = DynamicRepresentationLearning()
    framework = SelfEvolvingFramework()
    backend_selector = ContextAwareBackendSelector()
    runtime = RuntimeExecutionEngine()
    
    print(f"\n[INIT] Active Execution Engine: {runtime.device.upper()}")

    # 2. Simulate Incoming IoT Sensor Data
    print("\n[PHASE 1] Semantic Context & Representation Learning")
    sensor_data = np.random.randn(100, 8) # 8 feature dimensional data
    encoding = rep_learner.select_optimal_encoding(sensor_data, {})
    print(f"  -> Incoming Data Shape: {sensor_data.shape}")
    print(f"  -> Auto-Selected Encoding Strategy: {encoding}")

    # 3. Backend Selection based on Context
    print("\n[PHASE 2] Context-Aware Backend Selection")
    context = {
        'circuit_complexity': 30,  # qubits * depth
        'battery_level': 0.4,      # 40% battery on edge device
        'latency_critical': True,  # Needs fast response
        'sensor_noise_level': 0.05
    }
    selected_backend = backend_selector.select_optimal_backend(context)
    print(f"  -> IoT Device Context: Battery={context['battery_level']*100}%, LatencyCritical=True")
    print(f"  -> Selected Backend: {selected_backend.name} (Type: {selected_backend.qpu_type})")

    # 4. Evolution & Execution Loop
    print("\n[PHASE 3] Continuous Evolution & Execution")
    print(f"  -> Initial Quantum IR: {framework.current_ir.gates}")
    
    for step in range(1, 6):
        # Simulate environment drift at step 3
        accuracy = 0.92 if step < 3 else 0.65 
        
        print(f"\n  --- T={step} ---")
        # Execute the circuit through the C++/Python FFI Bridge
        state_vector = runtime.execute_circuit(framework.current_ir)
        print(f"  Execution complete. Output State Vector size: {len(state_vector)}")
        
        # Monitor performance
        framework.monitor_performance(accuracy=accuracy, latency=0.2)
        
        # Trigger evolution if drift is detected
        if framework.detect_drift():
            framework.trigger_evolution()
            framework.performance_history.clear()

    print("\n" + "="*60)
    print(" EXPERIMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_phd_simulation()
