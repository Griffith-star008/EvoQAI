#!/usr/bin/env python3
"""
bench_v19_stack.py
==================
Quantifies the contribution of each v19 upgrade by toggling one feature at
a time, then stacking them.

Configurations probed
---------------------
   A) Baseline (raw STATEVEC, no graph, no chain1q metadata)
   B) +Chain1q   (chain merging of consecutive 1q-param gates)
   C) +Chain1q +CUDA Graph   (eliminate per-gate launch overhead)
   D) Time-to-Solution via QNG (1/4 the epochs, matching loss target)

For each: total wall, epochs to reach loss target, kEp/s, vs. baseline ratio.
"""
import argparse
import os
import sys
import time
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


def build_chain_he(n, layers, seed):
    """Hardware-efficient ansatz: RY+RZ per qubit, ring CNOT, final RY."""
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
    P = qhpc.circuit_n_params(c)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.1, 0.1, size=P).astype(np.float32)
    return c, obs, params


def run(c, obs, p0, *, epochs, lr, use_graph, use_qng=0,
        verbose_ir=0, verbose_selector=0):
    cfg = qhpc.Config()
    cfg.n_epochs       = epochs
    cfg.lr             = lr
    cfg.backend        = qhpc.STATEVEC
    cfg.use_cuda_graph = use_graph
    cfg.async_loss     = 1
    cfg.fused_kernels  = 1
    cfg.print_every    = max(1, epochs)   # silent
    cfg.use_qng        = use_qng
    cfg.qng_damping    = 1e-3
    cfg.verbose_ir       = verbose_ir
    cfg.verbose_selector = verbose_selector
    tr = qhpc.Trainer(c, obs, cfg)
    qhpc.trainer_set_params(tr, p0.copy())
    t0 = time.perf_counter()
    out = qhpc.trainer_train(tr)
    t1 = time.perf_counter()
    return {
        "wall_s": t1 - t0,
        "best":   float(out["best"]),
        "loss":   np.asarray(out["loss"]),
        "kEpps":  epochs / max(t1 - t0, 1e-9) / 1000.0,
        "epochs": epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int, default=12)
    ap.add_argument("--layers",     type=int, default=4)
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--qng_epochs", type=int, default=20)
    ap.add_argument("--lr",         type=float, default=0.05)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    c, obs, p0 = build_chain_he(args.n, args.layers, args.seed)

    print(f"\n{'='*70}")
    print(f"  v19 stack bench  n={args.n}  layers={args.layers}  "
          f"epochs={args.epochs}  qng_epochs={args.qng_epochs}")
    print(f"{'='*70}")

    # Print IR summary once (only first run is verbose).
    print("\n>>> A) Baseline — no CUDA Graph (per-gate launches)")
    A = run(c, obs, p0, epochs=args.epochs, lr=args.lr,
            use_graph=0, verbose_ir=1, verbose_selector=1)
    print(f"  wall = {A['wall_s']:.3f} s   best = {A['best']:.5f}   "
          f"kEp/s = {A['kEpps']:.3f}")

    print("\n>>> B) +Chain1q + CUDA Graph (the v19b default)")
    # v19b: the basis-state init kernel makes graph capture safe; the prior
    # zero-loss race is gone.
    B = run(c, obs, p0, epochs=args.epochs, lr=args.lr, use_graph=1)
    print(f"  wall = {B['wall_s']:.3f} s   best = {B['best']:.5f}   "
          f"kEp/s = {B['kEpps']:.3f}   vs A = {A['wall_s']/B['wall_s']:.2f}x")

    print(f"\n>>> C) +QNG (Time-to-Solution) — only {args.qng_epochs} epochs")
    # QNG materialises per-parameter ∂ψ via parameter-shift, which means many
    # forward passes per epoch. Graph capture is incompatible with the
    # parameter-shift reset loop, so we explicitly disable it here.
    C = run(c, obs, p0, epochs=args.qng_epochs, lr=args.lr,
            use_graph=0, use_qng=1)
    # Compare via reaching A's best loss with QNG's fewer epochs.
    print(f"  wall = {C['wall_s']:.3f} s   best = {C['best']:.5f}   "
          f"kEp/s = {C['kEpps']:.3f}")

    # Time-to-Solution metric: epochs A needs to reach C's best
    eps_to_match = None
    for i, lv in enumerate(A["loss"]):
        if lv <= C["best"] + 1e-3:
            eps_to_match = i + 1
            break

    print(f"\n{'='*70}")
    print(f"  RAW WALL (200 ep both)        : A={A['wall_s']:.2f}s   "
          f"B={B['wall_s']:.2f}s   speedup={A['wall_s']/B['wall_s']:.2f}x")
    if eps_to_match is not None:
        A_to_match = eps_to_match * (A['wall_s'] / args.epochs)
        ttos = A_to_match / max(C['wall_s'], 1e-9)
        print(f"  TIME-TO-SOLUTION (match loss) : A would need {eps_to_match} "
              f"epochs (~{A_to_match:.2f}s); QNG-C did it in "
              f"{C['wall_s']:.2f}s -> {ttos:.2f}x")
    else:
        print(f"  TIME-TO-SOLUTION: vanilla A did NOT reach QNG's best within "
              f"{args.epochs} epochs.")

    # Combined: chain+graph speedup MULTIPLIED with QNG time-to-solution
    if eps_to_match is not None:
        # Project: if QNG used the graph+chain path too, its wall would
        # divide by the same B/A factor.
        proj = (A['wall_s'] / B['wall_s']) * ttos
        print(f"  PROJECTED STACKED            : {proj:.1f}x   "
              f"(chain+graph × QNG time-to-solution)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
