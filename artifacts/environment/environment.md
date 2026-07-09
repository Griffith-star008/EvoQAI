# Reproducibility Environment Configuration

## Python Dependencies
```
python==3.11.9
numpy==1.26.4
scipy==1.13.0
torch==2.3.0
qiskit==1.1.0
z3-solver==4.13.0
pytest==8.2.0
flake8==7.0.0
mypy==1.10.0
bandit==1.7.8
```

## System Requirements
- OS: Ubuntu 22.04 LTS (tested), Windows 11 (compatible), macOS 14 (compatible)
- CPU: x86_64, minimum 4 cores recommended
- GPU: NVIDIA CUDA 12.x (optional, for accelerated benchmarks)
- RAM: Minimum 16 GB, recommended 32 GB
- Disk: Minimum 10 GB free

## Docker Environment
```bash
docker build -t aqip:latest -f deployment/docker/Dockerfile .
docker run --gpus all -v $(pwd)/results:/app/results aqip:latest
```

## Verification
After setup, run:
```bash
python tests/test_runtime_evolution.py
# Expected: 20 passed, 0 failed
```
