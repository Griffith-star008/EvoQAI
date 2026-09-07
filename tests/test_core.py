import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from src.evoqai.quantum.vqc import AdaptiveVQC
from src.evoqai.world_model.causal_twin import CausalDigitalTwin
from src.evoqai.data.drift_generator import SEAStreamGenerator

def test_vqc_forward_pass():
    vqc = AdaptiveVQC(n_qubits=3, n_layers=1)
    x = torch.rand((4, 3)) # batch of 4, 3 features
    out = vqc(x)
    assert out.shape == (4,), "Output shape mismatch"

def test_vqc_mutation():
    vqc = AdaptiveVQC(n_qubits=3, n_layers=1)
    vqc.add_layer()
    assert vqc.n_layers == 2, "Mutation failed to add layer"

def test_causal_twin_safety():
    twin = CausalDigitalTwin(base_noise_rate=0.2, noise_threshold=0.5)
    # (1 - 0.2)^2 = 0.64 (Safe)
    assert twin.evaluate_mutation(current_layers=1, mutation_type="add_layer") == True
    # (1 - 0.2)^4 = 0.4096 (Unsafe)
    assert twin.evaluate_mutation(current_layers=3, mutation_type="add_layer") == False

def test_data_generator():
    stream = SEAStreamGenerator()
    x, y = stream.get_batch(10)
    assert x.shape == (10, 3)
    assert y.shape == (10,)
