# AQIP Operational Runbook

## 1. Normal Operations

### 1.1 Starting the Platform
```bash
# Single-node development mode
python -m aqip.runtime --mode=dev --config=artifacts/configs/benchmark_config.json

# Multi-node production mode (Docker Compose)
docker compose -f deployment/docker-compose.yml up -d

# Kubernetes deployment
helm install aqip deployment/helm/aqip/ --values deployment/helm/values-prod.yaml
```

### 1.2 Health Check
```bash
# Check runtime health
curl http://localhost:8080/health

# Expected response:
# {"status": "HEALTHY", "checks": [...], "uptime_seconds": 3600}
```

### 1.3 Monitoring
- **Metrics Dashboard:** `http://localhost:3000` (Grafana)
- **Structured Logs:** `logs/<component>_YYYYMMDD.jsonl`
- **Telemetry Export:** `http://localhost:9090/metrics` (Prometheus format)

---

## 2. Incident Response

### 2.1 Evolution Stall (no improvement for > 100 cycles)
**Symptoms:** AQII score plateaus, evolution counter increases but loss does not decrease.
**Diagnosis:**
```bash
grep "no improvement found" logs/runtime_evolution_*.jsonl | tail -20
```
**Resolution:**
1. Check if the system has reached a local optimum (expected behavior per Theorem 1).
2. If premature stall, increase `w_max` to allow larger subgraph permutations.
3. Reset the policy network to explore new permutation types.

### 2.2 Verification Timeout Spike
**Symptoms:** TIMEOUT rate exceeds 20% (normal: < 6%).
**Diagnosis:**
```bash
grep "TIMEOUT" logs/verifier_*.jsonl | wc -l
```
**Resolution:**
1. Reduce `w_max` (fewer qubits → faster verification).
2. Increase `timeout_ms` threshold.
3. Check for unusually large subgraphs being proposed.

### 2.3 Memory Pressure
**Symptoms:** OOM errors, health check reports DEGRADED.
**Resolution:**
1. Enable graph memory compression: `--enable-graph-compression`.
2. Reduce concurrent evolution proposals: `--max-concurrent-proposals=1`.
3. Scale horizontally by adding nodes to the cluster.

### 2.4 Rollback Procedure
```bash
# Checkpoint is taken before every evolution deployment
# List available checkpoints:
ls artifacts/checkpoints/

# Rollback to specific checkpoint:
python -m aqip.runtime --restore-checkpoint=artifacts/checkpoints/ckpt_20260709_120000.json
```

---

## 3. Deployment Procedures

### 3.1 Canary Deployment
1. Deploy new version to 1 node (canary).
2. Monitor AQII score and error rates for 30 minutes.
3. If canary is healthy, proceed to rolling update.
4. If canary is degraded, rollback immediately.

### 3.2 Rolling Update
```bash
kubectl rollout restart deployment/aqip-runtime -n aqip
kubectl rollout status deployment/aqip-runtime -n aqip --timeout=300s
```

### 3.3 Blue-Green Deployment
1. Deploy new version as "green" alongside existing "blue."
2. Run integration tests against green.
3. Switch load balancer from blue to green.
4. Retain blue for 24 hours as rollback target.

---

## 4. Security Procedures

### 4.1 Secret Rotation
```bash
# Rotate API keys every 90 days
kubectl create secret generic aqip-secrets --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
```

### 4.2 Dependency Audit
```bash
# Run weekly
pip audit
bandit -r src/ -ll
```

### 4.3 Artifact Integrity
```bash
# Verify checksums of all artifacts
python -c "
from src.research.tracking.experiment_logger import ExperimentLogger
# checksums are automatically verified on load
"
```
