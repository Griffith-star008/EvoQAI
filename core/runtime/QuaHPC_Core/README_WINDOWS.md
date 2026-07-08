# QHPC v16 — Windows / PowerShell quickstart

## TL;DR

```powershell
# In an elevated **x64 Native Tools Command Prompt for VS 2022** (or any
# PowerShell where cl.exe and nvcc.exe are on PATH), from this directory:

.\build_and_bench.ps1                       # default: sm_89, n=12, layers=4
.\build_and_bench.ps1 -SmArch 90            # Hopper (H100)
.\build_and_bench.ps1 -SmArch 86            # Ampere (RTX 30xx, A100)
.\build_and_bench.ps1 -N 14 -Layers 6 -Epochs 500
.\build_and_bench.ps1 -SkipBuild            # rerun bench only
.\build_and_bench.ps1 -SkipPennylane        # measure QHPC alone
```

The script does, in order:

1. **Preflight** — checks `cmake`, `nvcc`, `python`; auto-installs `pybind11`,
   `numpy`, `pennylane`, `pennylane-lightning-gpu` if missing.
2. **CMake configure** — Visual Studio 2022 generator, x64, SM arch passed
   through.
3. **Build** — `cmake --build` Release with parallelism.
4. **Stage** — copies `qhpc.pyd` (Python module) and `qhpc.dll` (core) next
   to the bench script, prepends `CUDA_PATH\bin` so the runtime DLLs
   (`cusolver64_*.dll`, `nvrtc64_*.dll`) are findable.
5. **Bench** — head-to-head VQE: identical ansatz + Hamiltonian + seed +
   epochs + Adam lr against `lightning.gpu`. Reports wall time, best loss,
   speedup ratio.

---

## What gets built

| Artifact          | What it is                                       |
| ----------------- | ------------------------------------------------ |
| `qhpc.pyd`        | Python extension module (the pybind11 wrapper)   |
| `qhpc.dll`        | Core shared library (C API + v16 CUDA backends)  |
| `build\`          | Out-of-source CMake tree                         |

Both DLLs live next to `bench_vs_pennylane.py` so Python's loader finds them
without extra `PATH` setup.

---

## Prerequisites (one-time)

- **Visual Studio 2022 Build Tools** (community/pro/enterprise) with
  "Desktop development with C++" workload. This gives you `cl.exe`.
- **CUDA Toolkit 12.x** — installer puts `nvcc.exe`, `cusolver64_*.dll`,
  `nvrtc64_*.dll` into `%CUDA_PATH%\bin`.
- **CMake** ≥ 3.20 — `winget install Kitware.CMake`.
- **Python** ≥ 3.10 — recommend the Microsoft Store or python.org installer.

The PS script will `pip install --user` pybind11/numpy/pennylane on demand,
so you don't need to pre-install them.

---

## Common failures and fixes

### `cl.exe not found`
Open **"x64 Native Tools Command Prompt for VS 2022"** → start PowerShell
from there:
```powershell
powershell.exe -NoExit
```
Or run `Launch-VsDevShell.ps1` from your VS install.

### `nvcc fatal: ... unsupported gpu architecture 'compute_89'`
Your CUDA toolkit is older than the SM arch you asked for. Either:
- Update to CUDA 12.4+ for Ada (sm_89) or 12.0+ for Hopper (sm_90), or
- Use the right arch flag: `.\build_and_bench.ps1 -SmArch 86` for Ampere.

### `ImportError: DLL load failed while importing qhpc`
Means the runtime DLL search couldn't find `qhpc.dll` or a CUDA runtime DLL.
Fix:
```powershell
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
python -c "import qhpc; print(qhpc.__version__, qhpc.BUILD_INFO)"
```
The PS script already prepends `CUDA_PATH\bin` before running the bench.

### `pennylane-lightning-gpu` install fails on Windows
PennyLane's GPU plugin currently has spotty Windows wheels. The bench falls
back to `lightning.qubit` (CPU) automatically and labels the result so the
speedup number stays interpretable. If you want a true GPU vs GPU number,
use WSL2 + Ubuntu and run the Linux build script instead.

### `nvcc warning: 'long_double' is treated as 'double'`
Harmless. Already silenced via `/wd4819 /wd4828` (codepage warnings on
Windows because some headers are UTF-8).

### `cuSOLVER call failed`
This usually means CUDA versions are mismatched between toolkit and driver.
Check:
```powershell
nvidia-smi             # driver version (right column)
nvcc --version         # toolkit version
```
Driver must be ≥ what the toolkit ships against. Update driver if needed.

---

## Reading the bench output

```
>>> QHPC v16 <<<
  wall_s = 0.412   best = -7.833214   params = 108   backend = 1
>>> PennyLane-Lightning-GPU (adjoint) <<<
  wall_s = 3.701   best = -7.833201   params = 108   device = lightning.gpu
========================================================
  Speedup       :  8.98x  (PennyLane / QHPC)
  Loss delta    :  +0.000013   (PL_best - QHPC_best)
  Correctness   :  OK (within 1e-3)
========================================================
```

- **Speedup** = PennyLane wall / QHPC wall. Includes startup. For epochs
  ≥ 500, dominated by per-epoch cost.
- **Loss delta** = how much QHPC's best loss deviates from PennyLane's.
  FP32 vs FP64 makes a small gap normal (≤ 1e-3). If it's larger, an
  ansatz mismatch is the most likely cause — verify the gate order.
- **backend** integer maps to: 0=STATEVEC_KERNELS, 1=NVRTC_MEGAKERNEL,
  2=HEISENBERG, 3=MPS, 4=DISTRIBUTED. For n=12 the AUTO selector should
  pick NVRTC (smem-resident megakernel).

---

## When to expect what

| Workload                       | QHPC backend  | Typical speedup vs PL-Lightning-GPU |
| ------------------------------ | ------------- | ----------------------------------- |
| n ≤ 13, dense ansatz           | NVRTC-MK      | 8-25x                               |
| n in [14, 30], moderate depth  | STATEVEC + TC | 3-8x                                |
| Clifford-heavy QAOA            | HEISENBERG    | 20-100x (it's a different complexity class) |
| n > 30                         | DISTRIBUTED   | scales with PE count (need NVSHMEM) |

100x universal speedup remains roadmap, not delivered. For the workloads
above the v16 numbers should be reproducible on Ada/Hopper.
