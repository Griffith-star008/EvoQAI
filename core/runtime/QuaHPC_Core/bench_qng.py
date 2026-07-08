#!/usr/bin/env python3
"""
bench_qng.py
============
Compare three runs on identical hardware-efficient VQE ansatz:

    A) Vanilla Adam, 200 epochs        (baseline)
    B) Adam + QNG (use_qng=1), 20 epochs
    C) Adam + QNG, target loss = A's final loss, report epochs needed

The QNG claim is *Time-to-Solution*, not pure throughput: QNG epochs cost
more (2P forward passes + cuBLAS Cgemm + cuSOLVER solve), but each epoch
moves much further on the loss surface.
"""
import argparse
import os
import sys
import time

import numpy as np

# Windows: explicit DLL search paths.
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


def build_circuit_obs(n, layers, seed):
    c = qhpc.Circuit(n=n)
    p = 0
    for _ in range(layers):
        for q in range(n):
            qhpc.circuit_ry(c, q, p); p += 1
        for q in range(n):
            qhpc.circuit_rz(c, q, p); p += 1
        for q in range(n - 1):
            qhpc.circuit_cnot(c, q, q + 1)
        if n > 2:
            qhpc.circuit_cnot(c, n - 1, 0)
    for q in range(n):
        qhpc.circuit_ry(c, q, p); p += 1
    obs = qhpc.Observable()
    for i in range(n - 1):
        qhpc.obs_add_term(obs, 0, 0, (1 << i) | (1 << (i+1)), 1.0)
    n_params = qhpc.circuit_n_params(c)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.1, 0.1, size=n_params).astype(np.float32)
    return c, obs, params


def run(c, obs, params0, *, epochs, use_qng, lr=0.05, qng_damping=1e-3):
    cfg = qhpc.Config()
    cfg.n_epochs = epochs
    cfg.lr = lr
    cfg.backend = qhpc.STATEVEC
    cfg.use_cuda_graph = 0
    cfg.async_loss = 1
    cfg.fused_kernels = 1
    cfg.print_every = max(1, epochs // 10)
    cfg.verbose_ir = 0
    cfg.verbose_selector = 0
    cfg.use_qng = 1 if use_qng else 0
    cfg.qng_damping = qng_damping
    tr = qhpc.Trainer(c, obs, cfg)
    qhpc.trainer_set_params(tr, params0)
    t0 = time.perf_counter()
    out = qhpc.trainer_train(tr)
    t1 = time.perf_counter()
    return {
        "wall_s": t1 - t0,
        "best":   float(out["best"]),
        "loss":   np.asarray(out["loss"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--vanilla_epochs", type=int, default=200)
    ap.add_argument("--qng_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    c, obs, p0 = build_circuit_obs(args.n, args.layers, args.seed)
    print(f"\n=== QNG bench | n={args.n}  layers={args.layers}  "
          f"params={len(p0)} ===\n")

    print(">>> A) Vanilla Adam   (use_qng=0)")
    A = run(c, obs, p0.copy(), epochs=args.vanilla_epochs, use_qng=False, lr=args.lr)
    print(f"  wall_s = {A['wall_s']:.3f}   best = {A['best']:.6f}   "
          f"epochs = {args.vanilla_epochs}\n")

    print(">>> B) Adam + QNG     (use_qng=1)")
    B = run(c, obs, p0.copy(), epochs=args.qng_epochs, use_qng=True, lr=args.lr)
    print(f"  wall_s = {B['wall_s']:.3f}   best = {B['best']:.6f}   "
          f"epochs = {args.qng_epochs}\n")

    # Speedup-to-same-loss: how many vanilla epochs needed to reach B's best?
    eps_to_match = None
    for i, lv in enumerate(A["loss"]):
        if lv <= B["best"] + 1e-3:
            eps_to_match = i + 1
            break

    print("=" * 58)
    print(f"  Vanilla best (200 ep) : {A['best']:.6f}   wall={A['wall_s']:.3f}s")
    print(f"  QNG     best ({args.qng_epochs:3d} ep) : {B['best']:.6f}   wall={B['wall_s']:.3f}s")
    if eps_to_match is not None:
        print(f"  Vanilla needed {eps_to_match} epochs to match QNG's best.")
        ttos = (eps_to_match * (A['wall_s'] / args.vanilla_epochs)
                ) / max(B['wall_s'], 1e-9)
        print(f"  Time-to-Solution speedup (QNG vs vanilla) : {ttos:.2f}x")
    else:
        print(f"  Vanilla did NOT reach QNG's best within {args.vanilla_epochs} epochs.")
    print("=" * 58)


if __name__ == "__main__":
    main()
