# Service Level Objectives (SLO) & Indicators (SLI)

To maintain production reliability, AQIP defines strict operational thresholds.

## 1. System Uptime
- **SLI:** Percentage of successful HTTP 200 responses to `/health` over a 30-day window.
- **SLO:** 99.9% (approx. 43 minutes of allowed downtime per month).
- **Consequence:** If breached, feature deployments are frozen until stability is restored.

## 2. Verification Reliability
- **SLI:** Percentage of SMT Verification calls that return either SAFE or UNSAFE (i.e., not TIMEOUT).
- **SLO:** 95.0% over a 24-hour rolling window.
- **Consequence:** If breached, the system automatically reduces $w_{max}$ by 2 to shrink the subgraph bound and ease solver pressure.

## 3. Evolution Latency
- **SLI:** P95 latency of a single evolution cycle (Propose $\to$ Verify $\to$ Deploy).
- **SLO:** < 60 ms over a 1-hour window.
- **Consequence:** If breached, scale out the Digital Twin simulator pool to distribute load.

## 4. Graph Integrity
- **SLI:** Number of invariant violations detected by the runtime graph monitor.
- **SLO:** Exactly 0.
- **Consequence:** A single violation triggers an immediate Pod restart and rollback to the last known-good checkpoint.
