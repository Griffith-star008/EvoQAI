import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.evoqai.quantum.vqc import AdaptiveVQC
from src.evoqai.data.drift_generator import SEAStreamGenerator
from src.evoqai.engine.adaptive_runner import AdaptiveEngine

# Classical MLP Baseline
class ClassicalMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.n_layers = 1 # Dummy for engine compatibility
        self.backend = "cpu"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Output in [-1, 1] to match VQC
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def run_single_pipeline(method_name, seed, epochs=150, batch_size=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    stream = SEAStreamGenerator(noise_percentage=0.05)
    
    # Initialize Models
    if method_name == "EvoQAI":
        model = AdaptiveVQC(n_qubits=3, n_layers=1)
        engine = AdaptiveEngine(model, lr=0.1, is_safe_mode=False) # Phase 1 focus
    elif method_name == "Static VQC":
        model = AdaptiveVQC(n_qubits=3, n_layers=1)
        # Override adapt to do nothing
        engine = AdaptiveEngine(model, lr=0.1)
        engine._adapt = lambda: None 
    elif method_name == "Static MLP":
        model = ClassicalMLP()
        engine = AdaptiveEngine(model, lr=0.05)
        engine._adapt = lambda: None
    elif method_name == "MLP + ADWIN(Retrain)":
        model = ClassicalMLP()
        engine = AdaptiveEngine(model, lr=0.05)
        # Override to reset weights on drift
        def reset_mlp():
            # Retrain from scratch
            engine.model = ClassicalMLP()
            engine.optimizer = optim.Adam(engine.model.parameters(), lr=0.05)
            engine.drift_detector.reset()
        engine._adapt = reset_mlp

    accuracies = []
    
    for epoch in range(epochs):
        if epoch == 75:
            stream.trigger_drift(new_threshold=14.0)
            
        x, y = stream.get_batch(batch_size)
        _, acc = engine.train_step(x, y)
        accuracies.append(acc)
        
    return accuracies

def run_all_baselines():
    methods = ["Static VQC", "Static MLP", "MLP + ADWIN(Retrain)", "EvoQAI"]
    seeds = [42, 123, 999]
    
    results = {m: [] for m in methods}
    
    print("Running Rigorous Baseline Comparisons (0 Overclaim)...")
    for method in methods:
        print(f"Evaluating {method}...")
        for seed in seeds:
            acc_history = run_single_pipeline(method, seed)
            # Smooth the accuracy
            smoothed = [sum(acc_history[i:i+10])/10 for i in range(len(acc_history)-10)]
            results[method].append(smoothed)
            
    # Calculate Mean and Std
    os.makedirs('experiments/reports', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    colors = {"Static VQC": "red", "Static MLP": "orange", "MLP + ADWIN(Retrain)": "green", "EvoQAI": "blue"}
    
    summary_data = {}
    
    for method in methods:
        arr = np.array(results[method]) # Shape: (seeds, epochs)
        mean_acc = np.mean(arr, axis=0)
        std_acc = np.std(arr, axis=0)
        
        epochs_range = range(len(mean_acc))
        plt.plot(epochs_range, mean_acc, label=method, color=colors[method])
        plt.fill_between(epochs_range, mean_acc - std_acc, mean_acc + std_acc, color=colors[method], alpha=0.1)
        
        summary_data[method] = {
            "mean_accuracy_post_drift": np.mean(mean_acc[75:]).item(),
            "recovery_time_epochs": int(np.argmax(mean_acc[75:] > 0.75)) if np.max(mean_acc[75:]) > 0.75 else -1
        }
        
    plt.axvline(x=75, color='black', linestyle='--', label='Concept Drift (SEA)')
    plt.xlabel('Streaming Batch (Epoch)')
    plt.ylabel('Accuracy (Mean ± Std over 3 seeds)')
    plt.title('Rigorous Baseline Comparison: Evolutionary QML vs Classical')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('experiments/reports/baseline_comparison.png')
    
    with open('experiments/reports/baseline_metrics.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print("\nBenchmark complete! Artifacts saved to experiments/reports/")
    print(json.dumps(summary_data, indent=4))

if __name__ == "__main__":
    run_all_baselines()
