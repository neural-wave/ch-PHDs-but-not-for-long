---
title: September-only original deterministic baselines
status: current
kind: experiment
recorded_at: '2026-10-16T21:39:09.391565+00:00'
sources:
- evaluate_september_baselines.py
- results/september_baseline_metrics.json
experiment_id: null
supersedes: []
---

User requested September-only errors for the three original methods. Evaluated unchanged saved baseline_predictions.csv for 2026-09-01 through 2026-09-30: 30 days, four zones, 11,520 unique quarters, 2,880 per zone. MAE EUR/MWh: hourly mean 10.69536631944445; linear interpolation 7.540344021267365; hourly-mean-corrected interpolation 7.090321234809023. Corrected reduction 33.7066069264%. Recomputed errors from predictions and cross-checked against 120 saved daily/zone MAEs per method to 1e-10. Same hourly/quarter centre alignment and constant daily boundaries; no fitting. Script evaluate_september_baselines.py; metrics results/september_baseline_metrics.json. Existing model work and job state were not modified or reverified.
