# Disaster Recovery Playbook

## 1. Scope
This playbook covers catastrophic failures where the UAIR graph becomes irrecoverably corrupted across multiple nodes, or a bad permutation escapes verification and enters the live production system.

## 2. Tier 1: State Corruption (Graph Invalid)
**Symptom:** Worker nodes crash with `GraphInvariantViolation` exceptions.
**Action:**
1. Isolate the affected deployment: `kubectl cordon <node>`
2. Fetch the last verified checkpoint:
   `kubectl exec -it <pod> -- python -m aqip.scripts.restore --checkpoint latest`
3. Restart the runtime engine with evolution temporarily disabled:
   `kubectl set env deployment/aqip-runtime EVOLUTION_ENABLED=false`

## 3. Tier 2: Solver Poisoning
**Symptom:** The SMT solver begins verifying permutations that are mathematically unsafe (due to a bug in the Z3 encoding).
**Action:**
1. Trigger the immediate Kill Switch: `curl -X POST http://aqip-control/killswitch`
2. Fallback to the purely static compiler (MLIR/Qiskit) via the API gateway route table.
3. Archive the `counterexamples.log` for offline forensic analysis.

## 4. Tier 3: Complete Cluster Loss
**Symptom:** The entire Kubernetes cluster is lost (e.g., cloud provider outage).
**Action:**
1. Provision a new cluster using the Terraform modules in `deployment/terraform/`.
2. Restore the most recent telemetry and checkpoint backup from AWS S3 / GCS.
3. Re-apply the Helm charts: `helm install aqip deployment/helm/aqip/`.
4. Estimated RTO (Recovery Time Objective): 15 minutes.
5. Estimated RPO (Recovery Point Objective): 1 evolution cycle (~50ms of lost work).
