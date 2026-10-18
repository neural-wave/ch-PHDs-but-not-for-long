---
title: Monthly comparison plot for original three baselines
status: current
kind: experiment
recorded_at: '2026-10-16T21:41:02.189088+00:00'
sources:
- results/monthly_baseline_mae.csv
- plot_monthly_baselines.py
experiment_id: null
supersedes: []
---

Aggregated results/baseline_daily_metrics.csv per month and zone, then equally averaged four zones. All 12 months/363 days validated; September agrees with independently computed September metrics within 1e-10. Mean-corrected interpolation has lowest MAE every month. September has highest and December lowest MAE for all methods. Outputs: results/monthly_baseline_mae.csv, results/monthly_baseline_mae.png, results/monthly_baseline_mae.svg. Reproduction: plot_monthly_baselines.py with matplotlib. Original baselines unchanged; no training or remote job launched.
