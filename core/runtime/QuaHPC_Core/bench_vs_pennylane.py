#!/usr/bin/env python3
"""
bench_vs_pennylane.py
=====================
Head-to-head VQE benchmark: QHPC v16 vs PennyLane-Lightning-GPU.

Both backends evaluate the SAME hardware-efficient ansatz on the SAME
Ising-Z Hamiltonian on a chain of N qubits, with the SAME number of
training epochs and the same Adam learning rate. Random initial
parameters are seeded identically.

Usage:
    python bench_vs_pennylane.py --n 12 --layers 4 --epochs 200

Requirements:
    pip install pennylane pennylane-lightning-gpu numpy
    qhpc:  built via the CMakeLists in this repo (libqhpc.so + qhpc.so)

Output:
    Wall-time + final loss for each backend, plus a speedup column.
"""
import argparse
import time
import sys
import os
import numpy as np

# --- Windows-only: Python 3.8+ ignored PATH for DLL search by default.
# We must explicitly add the directories where qhpc_core.dll + CUDA runtime
# DLLs (cudart64_*.dll, nvrtc64_*.dll, nvcuda.dll) live.
if sys.platform == "win32":
    _here = os.path.dirname(os.path.abspath(__file__))
    try:
        os.add_dll_directory(_here)
    except (OSError, AttributeError):
        pass
    _cuda = os.environ.get("CUDA_PATH")
    if _cuda:
        _cuda_bin = os.path.join(_cuda, "bin")
        if os.path.isdir(_cuda_bin):
            try:
                os.add_dll_directory(_cuda_bin)
            except (OSError, AttributeError):
                pass
    # System32 holds nvcuda.dll (driver). Should already be findable but add
    # explicitly to be safe.
    _sys32 = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32")
    if os.path.isdir(_sys32):
        try:
            os.add_dll_directory(_sys32)
        except (OSError, AttributeError):
            pass

try:
    import qhpc
    HAVE_QHPC = True
except ImportError as e:
    HAVE_QHPC = False
    print(f"[bench] qhpc not importable: {e}", file=sys.stderr)

try:
    import pennylane as qml
    HAVE_PL = True
except ImportError:
    HAVE_PL = False
    print("[bench] pennylane not installed; skipping that path", file=sys.stderr)


def build_chain_edges(n: int):
    return [(i, i + 1) for i in range(n - 1)]


def hamiltonian_ising_zz(n: int, J: float = 1.0):
    """Sum_{(i,j) in edges} J * Z_i Z_j. Z-only diagonal Hamiltonian."""
    return [(i, j, J) for (i, j) in build_chain_edges(n)]


# ---------------------------------------------------------------------------
# QHPC path
# ---------------------------------------------------------------------------
def run_qhpc(n: int, layers: int, epochs: int, lr: float, seed: int):
    if not HAVE_QHPC:
        return None

    # Hardware-efficient ansatz: per layer RY(q) + RZ(q) for each q, then ring
    # of CNOTs. Final RY layer. Same structure as PennyLane code below.
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
    assert qhpc.circuit_n_params(c) == p

    # Observable: sum_{(i,j)} Z_i Z_j  (z_mask = (1<<i)|(1<<j))
    obs = qhpc.Observable()
    for i, j, J in hamiltonian_ising_zz(n):
        qhpc.obs_add_term(obs, x_mask=0, y_mask=0, z_mask=(1 << i) | (1 << j), coeff=J)

    cfg = qhpc.Config()
    cfg.n_epochs = epochs
    cfg.lr = lr
    # v19b: CUDA Graph re-enabled. The basis-state initialiser used to be a
    # cudaMemcpyAsync from a host stack variable, which captured into the
    # graph as a reference to an invalid pointer at replay time and produced
    # zeroed state vectors. Replaced by a device kernel; graph is now safe.
    cfg.use_cuda_graph = 1
    cfg.async_loss = 1
    cfg.fused_kernels = 1
    cfg.print_every = max(1, epochs // 10)
    cfg.backend = qhpc.STATEVEC
    cfg.verbose_ir = 1
    cfg.verbose_selector = 1

    tr = qhpc.Trainer(c, obs, cfg)

    # Override with same seeded init params PennyLane uses, for fairness.
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(-0.1, 0.1, size=qhpc.trainer_n_params(tr)).astype(np.float32)
    qhpc.trainer_set_params(tr, p0)

    t0 = time.perf_counter()
    out = qhpc.trainer_train(tr)
    t1 = time.perf_counter()
    return {
        "loss":    np.asarray(out["loss"]),
        "wall_s":  t1 - t0,
        "wall_s_internal": float(out.get("wall_s", t1 - t0)),
        "best":    float(out.get("best", np.min(out["loss"]) if len(out["loss"]) else 0.0)),
        "backend": int(out.get("backend", -1)),
        "n_params": int(qhpc.trainer_n_params(tr)),
    }


# ---------------------------------------------------------------------------
# PennyLane path
# ---------------------------------------------------------------------------
def run_pennylane(n: int, layers: int, epochs: int, lr: float, seed: int,
                   diff: str = "adjoint", device: str = "lightning.gpu"):
    if not HAVE_PL:
        return None

    try:
        dev = qml.device(device, wires=n)
    except Exception as e:
        print(f"[bench] PennyLane device '{device}' not available: {e}", file=sys.stderr)
        # Fall back to lightning.qubit (CPU) so we still produce a number
        dev = qml.device("lightning.qubit", wires=n)
        device = "lightning.qubit"

    # Pauli observable matching QHPC obs.
    coeffs = [J for _, _, J in hamiltonian_ising_zz(n)]
    obs    = [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in build_chain_edges(n)]
    H = qml.Hamiltonian(coeffs, obs)

    n_params = 2 * n * layers + n  # match QHPC ansatz exactly

    def ansatz(params):
        p = 0
        for _ in range(layers):
            for q in range(n):
                qml.RY(params[p], wires=q); p += 1
            for q in range(n):
                qml.RZ(params[p], wires=q); p += 1
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
            if n > 2:
                qml.CNOT(wires=[n - 1, 0])
        for q in range(n):
            qml.RY(params[p], wires=q); p += 1

    @qml.qnode(dev, diff_method=diff)
    def cost(params):
        ansatz(params)
        return qml.expval(H)

    rng = np.random.default_rng(seed)
    # PennyLane requires its OWN numpy ndarray wrapper with requires_grad=True
    # for autograd to track parameters. A plain np.float64 array makes Adam
    # see grad=0 and the loss never moves.
    init = rng.uniform(-0.1, 0.1, size=n_params).astype(np.float64)
    params = qml.numpy.array(init, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    losses = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        params, loss = opt.step_and_cost(cost, params)
        losses.append(float(loss))
        if (ep % max(1, epochs // 10)) == 0:
            print(f"[PL] epoch {ep}/{epochs}  loss={loss:.6f}")
    t1 = time.perf_counter()

    return {
        "loss":    np.asarray(losses),
        "wall_s":  t1 - t0,
        "best":    float(min(losses)) if losses else float("inf"),
        "device":  device,
        "diff":    diff,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=12, help="qubits")
    ap.add_argument("--layers",  type=int, default=4)
    ap.add_argument("--epochs",  type=int, default=200)
    ap.add_argument("--lr",      type=float, default=0.05)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--skip_pl", action="store_true")
    ap.add_argument("--skip_qhpc", action="store_true")
    args = ap.parse_args()

    print(f"\n=== VQE benchmark | n={args.n} layers={args.layers} epochs={args.epochs} lr={args.lr} ===\n")

    qhpc_res = None
    if not args.skip_qhpc:
        print(">>> QHPC v16 <<<")
        qhpc_res = run_qhpc(args.n, args.layers, args.epochs, args.lr, args.seed)
        if qhpc_res:
            print(f"  wall_s = {qhpc_res['wall_s']:.3f}  best = {qhpc_res['best']:.6f}  "
                  f"params = {qhpc_res['n_params']}  backend = {qhpc_res['backend']}\n")

    pl_res = None
    if not args.skip_pl:
        print(">>> PennyLane-Lightning-GPU (adjoint) <<<")
        pl_res = run_pennylane(args.n, args.layers, args.epochs, args.lr, args.seed,
                                diff="adjoint", device="lightning.gpu")
        if pl_res:
            print(f"  wall_s = {pl_res['wall_s']:.3f}  best = {pl_res['best']:.6f}  "
                  f"params = {pl_res['n_params']}  device = {pl_res['device']}\n")

    if qhpc_res and pl_res:
        speedup = pl_res["wall_s"] / max(qhpc_res["wall_s"], 1e-9)
        loss_diff = pl_res["best"] - qhpc_res["best"]
        print("=" * 56)
        print(f"  Speedup       :  {speedup:.2f}x  (PennyLane / QHPC)")
        print(f"  Loss delta    :  {loss_diff:+.6f}  (PL_best - QHPC_best)")
        if abs(loss_diff) < 1e-3:
            print("  Correctness   :  OK (within 1e-3)")
        else:
            print("  Correctness   :  CHECK — losses differ; verify ansatz match")
        print("=" * 56)


if __name__ == "__main__":
    main()
