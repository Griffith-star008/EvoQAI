import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import os
from src.evoqai.quantum.vqc import AdaptiveVQC
from src.evoqai.data.drift_generator import SineStreamGenerator
from src.evoqai.world_model.causal_twin import CausalDigitalTwin
from src.evoqai.engine.adaptive_runner import AdaptiveEngine

def run_routing_experiment():
    print("Starting Phase 3: AIoT Context-Aware Backend Routing")
    
    n_qubits = 3
    # Use high base noise to trigger some mutations early if needed, but we mainly want to see routing
    model = AdaptiveVQC(n_qubits=n_qubits, n_layers=1, base_noise_rate=0.01)
    
    # We use Sine stream which is complex
    stream = SineStreamGenerator(noise_percentage=0.05)
    
    twin = CausalDigitalTwin(gate_error_rate=0.05, noise_threshold=0.55)
    
    # Enable safe mode
    engine = AdaptiveEngine(model, causal_twin=twin, lr=0.1, is_safe_mode=True)
    
    epochs = 120
    batch_size = 16
    
    history_battery = []
    history_backend = []
    history_depth = []
    
    for epoch in range(epochs):
        # Trigger drift at epoch 60 to increase complexity
        if epoch == 60:
            print(f"--- Epoch {epoch}: Concept Drift Triggered (Phase Shift) ---")
            stream.trigger_drift(new_phase_shift=2.0)
            
        x, y = stream.get_batch(batch_size)
        loss, acc = engine.train_step(x, y)
        
        history_battery.append(engine.simulated_battery)
        history_backend.append(engine.model.backend)
        history_depth.append(engine.model.n_layers)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Acc: {acc:.2f} | Battery: {engine.simulated_battery:.2f} | Backend: {engine.model.backend} | Depth: {engine.model.n_layers}")
            
    # Save Results
    os.makedirs("experiments/reports", exist_ok=True)
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(history_battery, label='Battery Level', color='green')
    ax1.set_ylabel('Battery')
    ax1.legend()
    ax1.grid(True)
    
    # Map backends to Y-axis integers for plotting
    backend_map = {"default.qubit": 0, "qiskit.aer": 1, "qiskit.ibmq": 2}
    y_backends = [backend_map[b] for b in history_backend]
    
    ax2.step(range(epochs), y_backends, label='Backend Selection', color='blue', where='post')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Edge (Local)', 'Cloud Sim', 'Cloud QPU'])
    ax2.set_ylabel('Backend')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(history_depth, label='Circuit Depth', color='red')
    ax3.set_ylabel('Layers')
    ax3.set_xlabel('Epochs')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("experiments/reports/aiot_routing_behavior.png")
    print("\nRouting experiment complete! Plot saved to experiments/reports/aiot_routing_behavior.png")

if __name__ == "__main__":
    run_routing_experiment()
