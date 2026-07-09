# Operational Readiness Review (ORR) Checklist

Before deploying any new Major or Minor version of AQIP to a production cluster, the infrastructure team must sign off on this ORR checklist.

## 1. Architecture & Design
- [ ] Has the Service Dependency Map (`service_dependencies.md`) been updated?
- [ ] Are all new APIs documented in the OpenAPI specification?
- [ ] Have all architectural changes been formalized in an ADR?

## 2. Security & Compliance
- [ ] Has `pip audit` / `bandit` been run with zero High/Critical vulnerabilities?
- [ ] Is the `SBOM.json` up to date and verified against the container image?
- [ ] Are all new configuration secrets stored securely in Vault (not hardcoded)?
- [ ] Have the SMT verifier bounds ($w_{max}$) been audited against hardware limits?

## 3. Capacity & Cost
- [ ] Have the Resource Requests and Limits (CPU/Memory) been updated in the Helm charts?
- [ ] Is the Digital Twin configured to run on Spot instances (Cost Optimization)?
- [ ] Has the performance budget (`performance_budgets.json`) been validated in CI?

## 4. Observability & Alerting
- [ ] Are SLIs for the new features reporting correctly to Prometheus?
- [ ] Have Grafana dashboards been updated to reflect new node types?
- [ ] Do alerts route to the correct PagerDuty escalation policies?

## 5. Resilience & Recovery
- [ ] Has the Disaster Recovery Playbook been tested for this version (e.g., restoring a checkpoint)?
- [ ] Does the Canary Deployment pipeline correctly rollback if the `aqii_score` SLI drops by >10%?
- [ ] Are Graceful Shutdowns implemented to prevent interrupting long-running QPU jobs?
