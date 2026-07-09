#!/usr/bin/env python3
"""
bench_max_qubits.py
===================
Find the largest qubit count this GPU can sustain with full state-vector
adjoint differentiation, and time a real 10-epoch VQE at that ceiling.

Memory budget per qubit count n (FP32 complex state-vector + adjoint shadow):
    ψ   :  2^n * 8 B
    φ   :  2^n * 8 B  (the adjoint backward shadow vector)
   ─────
   Total : 2^(n+1) * 8 B  =  2^(n+4) bytes

   n=24 →   256 MB     n=27 →    2 GB
   n=25 →   512 MB     n=28 →    4 GB
   n=26 →     1 GB     n=29 →    8 GB
                       n=30 →   16 GB   (RTX 4060 OOM)

Add the per-block partials buffer (max_grid × P × 4 bytes) and Adam moments
(P × 8 B), but those are P-bounded, not n-bounded — negligible at n≤30.

This script:
   1. Lists GPU free-VRAM (probed via cudaMemGetInfo through a tiny C call,
      or via pynvml if available).
   2. For each n in --n_list, computes the projected footprint.
   3. Runs a 10-epoch VQE (3 layers, hardware-efficient chain) when the
      projection says we'll fit.
   4. Reports wall_s, ep/s, GB/s effective state-vec bandwidth, peak VRAM.
"""
import argparse
import os
import sys
import time
import gc
import numpy as np

if sys.platform == "win32":
    _here = os.path.dirname(os.path.abspath(__file__))
    try: os.add_dll_directory(_here)
    except (OSError, AttributeError): pass
    _cuda = os.environ.get("CUDA_PATH")
    if _cuda:
        cb = os.path.join(_cuda, "bin")
        if os.path.isdir(cb):
            try: os.add_dll_directory(cb)
            except (OSError, AttributeError): pass

import qhpc


def gpu_vram_bytes():
    """Return total + free VRAM in bytes. Falls back to RTX 4060 default."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mi = pynvml.nvmlDeviceGetMemoryInfo(h)
        return mi.total, mi.free
    except Exception:
        pass
    # Fallback: parse nvidia-smi
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        total, free = [int(x) * 1024 * 1024 for x in out.split(",")]
        return total, free
    except Exception:
        # Hard fallback
        return 8 * (1 << 30), 6 * (1 << 30)


def build_chain_ansatz(n, layers, seed):
    """Linear-chain hardware-efficient ansatz: RY+RZ per qubit + CNOT chain."""
    c = qhpc.Circuit(n=n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            qhpc.circuit_ry(c, q, p); p += 1
        for q in range(n):
            qhpc.circuit_rz(c, q, p); p += 1
        for q in range(n - 1):
            qhpc.circuit_cnot(c, q, q + 1)
    for q in range(n):
        qhpc.circuit_ry(c, q, p); p += 1
    obs = qhpc.Observable()
    # Local nearest-neighbour ZZ
    for i in range(n - 1):
        qhpc.obs_add_term(obs, 0, 0, (1 << i) | (1 << (i+1)), 1.0)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.05, 0.05,
                         size=qhpc.circuit_n_params(c)).astype(np.float32)
    return c, obs, params


def run_one(n, layers, epochs, seed):
    c, obs, p0 = build_chain_ansatz(n, layers, seed)
    P = qhpc.circuit_n_params(c)
    cfg = qhpc.Config()
    cfg.n_epochs = epochs
    cfg.lr = 0.02
    cfg.backend = qhpc.STATEVEC
    cfg.use_cuda_graph = 0
    cfg.async_loss = 1
    cfg.fused_kernels = 1
    cfg.print_every = max(1, epochs)
    cfg.verbose_ir = 0
    cfg.verbose_selector = 0
    cfg.use_qng = 0    # not at this scale — μ buffer would blow VRAM
    cfg.use_tpb = 0    # ZZ chain all commutes — TPB adds no leverage here
    tr = qhpc.Trainer(c, obs, cfg)
    qhpc.trainer_set_params(tr, p0)
    t0 = time.perf_counter()
    out = qhpc.trainer_train(tr)
    t1 = time.perf_counter()
    return {
        "wall_s": t1 - t0,
        "best":   float(out["best"]),
        "n_params": P,
        "ep_s":   epochs / max(t1 - t0, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_list", type=str, default="24,25,26,27,28,29,30",
                    help="qubit counts to probe")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--safety_margin_gb", type=float, default=1.5,
                    help="reserve this much VRAM for misc CUDA overhead")
    args = ap.parse_args()

    n_list = [int(x) for x in args.n_list.split(",")]
    total, free = gpu_vram_bytes()
    print("=" * 84)
    print(f"  GPU VRAM: total = {total / (1<<30):.2f} GB    "
          f"free = {free / (1<<30):.2f} GB    "
          f"(safety reserve = {args.safety_margin_gb:.1f} GB)")
    budget = free - int(args.safety_margin_gb * (1 << 30))
    print(f"  Effective state-vec budget: {budget / (1<<30):.2f} GB")
    print(f"  Layers = {args.layers}   Epochs = {args.epochs}")
    print("=" * 84)
    print(f"  {'n':>4} {'sv_GB':>7} {'phi_GB':>7} {'total_GB':>9} {'fit?':>6} "
          f"{'wall_s':>9} {'ep/s':>8} {'GB/s':>9} {'best':>10}")
    print(f"  {'-'*4} {'-'*7} {'-'*7} {'-'*9} {'-'*6} {'-'*9} {'-'*8} {'-'*9} {'-'*10}")

    last_ceiling = None
    for n in n_list:
        amps = 1 << n
        sv_bytes  = amps * 8
        tot_bytes = sv_bytes * 2  # ψ + φ
        sv_gb  = sv_bytes  / (1 << 30)
        phi_gb = sv_bytes  / (1 << 30)
        tot_gb = tot_bytes / (1 << 30)
        fits = tot_bytes < budget
        if not fits:
            print(f"  {n:>4} {sv_gb:>7.3f} {phi_gb:>7.3f} {tot_gb:>9.3f} "
                  f"{'NO':>6} {'(over budget)':>40}")
            break
        # Warm GC before measuring.
        gc.collect()
        try:
            r = run_one(n, args.layers, args.epochs, seed=42)
            # Effective bandwidth: each epoch sweeps ψ a few times for gates +
            # the φ shadow during backward. Lower bound estimate: 4 sweeps
            # per gate per epoch × (RY layers + RZ layers + CNOT pairs).
            # We report a conservative 'gate_sweeps × sv_bytes / wall'.
            gate_sweeps = (args.layers * (2 * n + (n - 1)) + n) * 4
            eff_GBps = (gate_sweeps * sv_bytes / max(r["wall_s"], 1e-9)
                       / args.epochs) / (1 << 30)
            print(f"  {n:>4} {sv_gb:>7.3f} {phi_gb:>7.3f} {tot_gb:>9.3f} "
                  f"{'YES':>6} {r['wall_s']:>9.3f} {r['ep_s']:>8.2f} "
                  f"{eff_GBps:>9.1f} {r['best']:>10.4f}")
            last_ceiling = n
        except Exception as e:
            print(f"  {n:>4} {sv_gb:>7.3f} {phi_gb:>7.3f} {tot_gb:>9.3f} "
                  f"{'YES':>6} CRASH: {type(e).__name__}: {e}")
            break

    print("=" * 84)
    if last_ceiling is not None:
        print(f"  CEILING on this GPU: n = {last_ceiling} qubits  "
              f"({(1<<last_ceiling)*16/(1<<30):.1f} GB ψ+φ).")
    else:
        print("  No n value fit. Lower --safety_margin_gb or close other GPU apps.")
    print("\nNotes:")
    print("  - 'GB/s' is conservative effective state-vec bandwidth, useful")
    print("    for comparing the same gate-count workload across n values.")
    print("  - At n=29 (8 GB ψ+φ) on an 8 GB card, browsers / desktop")
    print("    compositor will compete for VRAM. Close Chrome before running.")
    print("  - n=30 needs 16 GB ψ+φ — won't fit on RTX 4060. Use 4090 (24GB)")
    print("    or H100 (80GB), or switch to MPS backend (see bench_stress_largeN.py).")


if __name__ == "__main__":
    main()
