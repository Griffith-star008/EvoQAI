# Security Policy

## 1. Threat Model
AQIP operates in environments with the following trust boundaries:
- **Trusted:** The UAIR compiler, SMT verifier, and runtime evolution engine.
- **Semi-trusted:** User-submitted workloads and custom optimization passes.
- **Untrusted:** External network traffic, third-party dependencies.

## 2. Access Control
| Resource | Policy |
|:---|:---|
| UAIR Graph Modification | Only the verified evolution engine may modify the graph |
| Configuration Changes | Requires authenticated API call with RBAC token |
| Benchmark Results | Read-only for all users, write for automation only |
| Secret Storage | Kubernetes Secrets / HashiCorp Vault |

## 3. Runtime Integrity
- Every structural permutation is verified by the SMT solver before deployment (Theorem 3).
- The evolution engine maintains an audit log of all accepted and rejected permutations.
- Digital Twin simulations run in an isolated sandbox; they cannot affect the live graph.

## 4. Dependency Verification
```bash
# All Python dependencies are pinned with exact versions
pip install -r requirements.txt --require-hashes

# Weekly automated scan
pip audit --strict
```

## 5. Audit Logging
All security-relevant events are logged in structured JSON format:
```json
{
  "timestamp": "2026-07-09T12:00:00Z",
  "event": "PERMUTATION_DEPLOYED",
  "component": "evolution_engine",
  "permutation_hash": "sha256:abc123...",
  "verification_result": "SAFE",
  "actor": "system/auto-evolution"
}
```

## 6. Vulnerability Disclosure
- Report security vulnerabilities to: security@aqip-project.org
- Do NOT report via public issue tracker.
- Expected response time: 72 hours.
- We follow coordinated disclosure (90-day window).
