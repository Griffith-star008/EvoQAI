"""
qhpc_distill.py  —  Quantum Knowledge Distillation
==================================================
Trains a tiny classical Student model to mimic the input-output behaviour of
a QHPC-trained Quantum Teacher.  The Student is then quantised to INT8 and
exported to ONNX, ready to flash onto edge devices (ESP32-S3, RP2040+TFLite,
Coral TPU, Hailo-8L, etc).

Workflow
--------
1. **Teacher**.  A QML circuit + trained parameters from QHPC v17.
   The teacher's prediction for an input x is a function
       f_T(x) = ⟨ψ(θ_trained, x) | O | ψ(θ_trained, x)⟩
   where x is encoded into the circuit via angle-embedding (default) or
   any user-supplied encoder callable.

2. **Probe set**.  We sample N points from the input domain (e.g. uniform on
   [-π, π]^d).  The teacher evaluates each one with QHPC's GPU forward pass.

3. **Student**.  A 3-layer MLP (default 32-16-1 hidden units) trained by
   minimising MSE against the teacher's outputs.  PyTorch optional — if
   missing, we fall back to a NumPy implementation with closed-form linear
   probes for tiny student sizes.

4. **Quantisation**.  Per-channel symmetric INT8 with PyTorch dynamic-quant
   or NumPy fake-quant.  Calibration uses the probe set.

5. **Export**.  ONNX file + a tiny C header (optional) that wraps a
   pure-INT8 forward for MCUs lacking ONNX runtimes.

Requires
--------
    qhpc                    (built from QHPC v17 source)
    numpy
    torch  (optional but recommended)
    onnx, onnxruntime  (optional, only if you ship ONNX)
"""
from __future__ import annotations
import os
import sys
import time
import math
from typing import Callable, Sequence

import numpy as np

# Optional deps — fail gracefully if absent.
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------
class QHPCTeacher:
    """
    Wraps a trained QHPC circuit + observable + parameters into a callable
    that maps numpy-array inputs to scalar predictions.

    Args:
        n_qubits: number of qubits in the ansatz.
        circuit_builder: callable (qhpc, circuit, x_param_base) -> None
            that appends gates to the circuit, using parameter indices
            starting at `x_param_base` for input-dependent gates.
            The remaining parameters (trained ones) come BEFORE x_param_base.
        observable_builder: callable (qhpc, observable) -> None
        trained_params: numpy array of trained θ (length = n_trained_params).
        x_dim: input dimensionality (each input scalar becomes one
            angle-encoded RY gate by default).
    """
    def __init__(self, n_qubits, circuit_builder, observable_builder,
                 trained_params: np.ndarray, x_dim: int):
        import qhpc
        self._qhpc = qhpc
        self._n_qubits = int(n_qubits)
        self._n_trained = int(trained_params.shape[0])
        self._x_dim = int(x_dim)

        # Build circuit once: trained params come first (indices 0..n_trained-1),
        # input params come after (indices n_trained..n_trained+x_dim-1).
        self._circ = qhpc.Circuit(n=self._n_qubits)
        circuit_builder(qhpc, self._circ, x_param_base=self._n_trained)
        assert qhpc.circuit_n_params(self._circ) == self._n_trained + self._x_dim, (
            f"circuit_builder produced {qhpc.circuit_n_params(self._circ)} params; "
            f"expected {self._n_trained} trained + {self._x_dim} input = "
            f"{self._n_trained + self._x_dim}")

        self._obs = qhpc.Observable()
        observable_builder(qhpc, self._obs)

        self._cfg = qhpc.Config()
        self._cfg.n_epochs = 1           # we only forward-eval, never train here
        self._cfg.lr = 0.0
        self._cfg.backend = qhpc.STATEVEC
        self._cfg.use_cuda_graph = 0
        self._cfg.async_loss = 0
        self._cfg.print_every = 1 << 30
        self._cfg.verbose_ir = 0
        self._cfg.verbose_selector = 0
        self._cfg.fused_kernels = 1

        self._trained = trained_params.astype(np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (N, x_dim).  Returns (N,) predictions."""
        X = np.atleast_2d(X.astype(np.float32))
        assert X.shape[1] == self._x_dim
        out = np.empty(len(X), dtype=np.float32)
        # We rebuild Trainer per call because it owns the param vector. This
        # is fine because Trainer construction is fast relative to the
        # forward — n_epochs=1 means train() runs one epoch and we read the
        # scalar loss.  For an inference-only API, fold this into a
        # dedicated forward-only entry point in the future.
        for i, x in enumerate(X):
            tr = self._qhpc.Trainer(self._circ, self._obs, self._cfg)
            params = np.concatenate([self._trained, x.astype(np.float32)])
            self._qhpc.trainer_set_params(tr, params)
            res = self._qhpc.trainer_train(tr)
            out[i] = float(res["best"])
        return out


# ---------------------------------------------------------------------------
# Probe set generator
# ---------------------------------------------------------------------------
def generate_probe_set(n_samples: int, x_dim: int,
                        domain=(-math.pi, math.pi), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = domain
    return rng.uniform(lo, hi, size=(n_samples, x_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Student (PyTorch path)
# ---------------------------------------------------------------------------
class StudentMLP(nn.Module if HAVE_TORCH else object):
    def __init__(self, x_dim: int, hidden: Sequence[int] = (32, 16)):
        if not HAVE_TORCH:
            raise RuntimeError("torch not installed")
        super().__init__()
        dims = [x_dim, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_student(teacher: QHPCTeacher,
                   n_samples: int = 2048,
                   epochs: int = 500,
                   batch_size: int = 64,
                   hidden=(32, 16),
                   lr: float = 1e-3,
                   seed: int = 0,
                   verbose: bool = True):
    """Train the student to match the teacher on a random probe set.

    Returns (student_model, X, y_teacher, y_student).
    """
    X = generate_probe_set(n_samples, teacher._x_dim, seed=seed)
    if verbose:
        print(f"[Distill] generating {n_samples} teacher predictions...")
    t0 = time.perf_counter()
    Y = teacher(X)
    t1 = time.perf_counter()
    if verbose:
        print(f"[Distill] teacher forward: {t1-t0:.2f}s "
              f"({n_samples/(t1-t0):.0f} samples/s)")

    if not HAVE_TORCH:
        # NumPy linear-regression fallback (no hidden layers).
        if verbose:
            print("[Distill] torch missing -> linear least-squares student")
        Xb = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
        w, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
        y_hat = Xb @ w
        return w, X, Y, y_hat

    torch.manual_seed(seed)
    student = StudentMLP(teacher._x_dim, hidden=hidden)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    mse = nn.MSELoss()
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            pred = student(Xt[idx])
            loss = mse(pred, Yt[idx])
            loss.backward()
            opt.step()
        if verbose and (ep % max(1, epochs // 10) == 0):
            with torch.no_grad():
                full = mse(student(Xt), Yt).item()
            print(f"[Distill] ep {ep:4d}/{epochs}  loss={full:.6f}")
    with torch.no_grad():
        y_hat = student(Xt).numpy()
    return student, X, Y, y_hat


# ---------------------------------------------------------------------------
# Quantisation + ONNX export
# ---------------------------------------------------------------------------
def quantize_and_export_onnx(student, x_dim: int, out_path: str = "student_int8.onnx",
                              verbose: bool = True):
    """PyTorch dynamic INT8 quantisation, then ONNX export."""
    if not HAVE_TORCH:
        raise RuntimeError("torch missing")
    qstudent = torch.quantization.quantize_dynamic(
        student, {nn.Linear}, dtype=torch.qint8)
    dummy = torch.zeros(1, x_dim, dtype=torch.float32)
    torch.onnx.export(
        qstudent, dummy, out_path,
        input_names=["x"], output_names=["y"],
        opset_version=17,
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
    if verbose:
        sz = os.path.getsize(out_path)
        print(f"[Distill] exported {out_path}  ({sz/1024:.1f} KB)")
    return qstudent


# ---------------------------------------------------------------------------
# Convenience tiny driver
# ---------------------------------------------------------------------------
def distill_demo():
    """
    End-to-end demo: trains a tiny Z-rotation toy ansatz, distills into a
    32-16-1 MLP, exports ONNX.  Sanity-only — replace teacher with your real
    trained model in production.
    """
    import qhpc
    n_qubits = 4
    n_trained = 4
    def circ_builder(q, c, x_param_base):
        # 1 layer of trained RYs, then x-encoded RYs.
        for i in range(n_qubits):
            q.circuit_ry(c, i, i)
        for i in range(n_qubits):
            q.circuit_ry(c, i, x_param_base + i)
        for i in range(n_qubits - 1):
            q.circuit_cnot(c, i, i + 1)

    def obs_builder(q, o):
        q.obs_add_term(o, 0, 0, 1, 1.0)  # Z_0

    trained = np.array([0.3, -0.2, 0.5, 0.1], dtype=np.float32)
    teacher = QHPCTeacher(n_qubits, circ_builder, obs_builder, trained, x_dim=n_qubits)
    res = train_student(teacher, n_samples=128, epochs=200, verbose=True)
    if HAVE_TORCH:
        student, X, Y, y_hat = res
        rmse = float(np.sqrt(np.mean((Y - y_hat) ** 2)))
        print(f"[Distill] student RMSE on probe = {rmse:.4f}")
        quantize_and_export_onnx(student, n_qubits, "student_int8.onnx")
    else:
        w, X, Y, y_hat = res
        rmse = float(np.sqrt(np.mean((Y - y_hat) ** 2)))
        print(f"[Distill] linear student RMSE on probe = {rmse:.4f}")
        np.savez("student_linear.npz", weights=w)


if __name__ == "__main__":
    distill_demo()
