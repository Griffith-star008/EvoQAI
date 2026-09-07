import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
from src.evoqai.quantum.vqc import AdaptiveVQC
from src.evoqai.data.drift_generator import SEAStreamGenerator
from src.evoqai.engine.adaptive_runner import AdaptiveEngine
from src.evoqai.world_model.causal_twin import CausalDigitalTwin

def run_pipeline(is_safe_mode: bool, base_noise=0.15, threshold=0.55):
    n_qubits = 3
    model = AdaptiveVQC(n_qubits=n_qubits, n_layers=1, base_noise_rate=base_noise)
    stream = SEAStreamGenerator(noise_percentage=0.05)
    twin = CausalDigitalTwin(base_noise_rate=base_noise, noise_threshold=threshold)
    engine = AdaptiveEngine(model, causal_twin=twin, lr=0.1, window_size=20, is_safe_mode=is_safe_mode)
    
    epochs = 150
    batch_size = 16
    accuracies = []
    
    for epoch in range(epochs):
        # Trigger continuous drifts
        if epoch == 50:
            stream.trigger_drift(new_threshold=11.0)
        if epoch == 100:
            stream.trigger_drift(new_threshold=14.0)
            
        x, y = stream.get_batch(batch_size)
        loss, acc = engine.train_step(x, y)
        accuracies.append(acc)
        
    return [sum(accuracies[i:i+10])/10 for i in range(len(accuracies)-10)]

def run_experiment():
    print("Starting Phase 2: Causal Digital Twin vs Unsafe Evolution")
    
    print("\n--- Running UNSAFE Evolution (Naive RL) ---")
    unsafe_acc = run_pipeline(is_safe_mode=False)
    
    print("\n--- Running SAFE Evolution (Causal Digital Twin) ---")
    safe_acc = run_pipeline(is_safe_mode=True)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(unsafe_acc)), unsafe_acc, color='red', label='Unsafe Evolution (Noise Collapse)')
    plt.plot(range(len(safe_acc)), safe_acc, color='green', label='Safe Evolution (Causal Twin Guided)')
    
    plt.axvline(x=50, color='gray', linestyle='--', label='Drift 1')
    plt.axvline(x=100, color='gray', linestyle='--', label='Drift 2')
    
    plt.xlabel('Streaming Batch (Epoch)')
    plt.ylabel('Accuracy')
    plt.title('EvoQAI: Causal Twin prevents Hardware Noise Collapse')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('experiments/reports', exist_ok=True)
    plt.savefig('experiments/reports/causal_twin_adaptation.png')
    print("\nExperiment complete! Plot saved to experiments/reports/causal_twin_adaptation.png")

if __name__ == "__main__":
    run_experiment()
