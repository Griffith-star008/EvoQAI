import matplotlib.pyplot as plt
from src.evoqai.quantum.vqc import AdaptiveVQC
from src.evoqai.data.drift_generator import SEAStreamGenerator
from src.evoqai.engine.adaptive_runner import AdaptiveEngine
import os

def run_experiment():
    print("Starting EvoQAI Phase 1: Evolutionary Quantum Circuit Adaptation under Concept Drift")
    
    # Initialize components
    n_qubits = 3
    model = AdaptiveVQC(n_qubits=n_qubits, n_layers=1)
    stream = SEAStreamGenerator(noise_percentage=0.05)
    engine = AdaptiveEngine(model, lr=0.1, window_size=20)
    
    epochs = 150
    batch_size = 16
    
    accuracies = []
    circuit_depths = []
    
    for epoch in range(epochs):
        # Trigger sudden concept drift at epoch 75
        if epoch == 75:
            print(f"\n[Epoch 75] TRIGGERING SUDDEN CONCEPT DRIFT")
            stream.trigger_drift(new_threshold=14.0)
            
        x, y = stream.get_batch(batch_size)
        loss, acc = engine.train_step(x, y)
        
        accuracies.append(acc)
        circuit_depths.append(model.n_layers)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {acc:.2f} | Layers: {model.n_layers}")
            
    # Smoothing for plot
    smoothed_acc = [sum(accuracies[i:i+10])/10 for i in range(len(accuracies)-10)]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Streaming Batch (Epoch)')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(range(len(smoothed_acc)), smoothed_acc, color=color, label='VQC Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=75, color='r', linestyle='--', label='Concept Drift (Epoch 75)')
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Circuit Depth (Layers)', color=color)
    ax2.plot(range(epochs), circuit_depths, color=color, linestyle=':', label='Circuit Depth')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    os.makedirs('experiments/reports', exist_ok=True)
    plt.title("EvoQAI: Quantum Circuit Evolution under Concept Drift")
    plt.savefig('experiments/reports/drift_adaptation.png')
    print("\nExperiment complete! Plot saved to experiments/reports/drift_adaptation.png")

if __name__ == "__main__":
    run_experiment()
