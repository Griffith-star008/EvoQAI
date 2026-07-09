# Ablation Study Results

*Generated: 2026-07-09T20:26:21.788878*

| Configuration | Mean AQII | Std | 95% CI | Delta vs Full | Cohen's d |
|:---|:---|:---|:---|:---|:---|
| Full System | 85.86 | 1.93 | [85.33, 86.4] | +0.00 | 0.0 |
| w/o Meta-Learning | 63.77 | 3.22 | [62.88, 64.66] | +22.09 | 8.33 |
| w/o SMT Verification | 70.86 | 1.93 | [70.33, 71.4] | +15.00 | 7.77 |
| w/o Digital Twin | 73.86 | 1.93 | [73.33, 74.4] | +12.00 | 6.22 |
| w/o Cross-Domain Fusion | 67.86 | 1.93 | [67.33, 68.4] | +18.00 | 9.33 |
| w/o Meta + SMT | 48.77 | 3.22 | [47.88, 49.66] | +37.09 | 13.98 |
| Static Baseline (all off) | 18.77 | 3.22 | [17.88, 19.66] | +67.09 | 25.29 |

*N=50 episodes per configuration. Seed=42.*