---
title: Verified initial baseline reconstruction results
status: current
kind: experiment
recorded_at: '2026-10-16T18:33:55.071545+00:00'
sources:
- results/BASELINES.md
- results/baseline_metrics.json
- evaluate_baselines.py
experiment_id: null
supersedes: []
---

Imported the three requested files and evaluated 139,392 quarters across 363 days/four zones. See results/BASELINES.md and results/baseline_metrics.json for source-linked details. MAE EUR/MWh: hourly mean 7.577610443927916; linear interpolation 5.738218670189106; hourly-mean-corrected interpolation 5.474192600723139. Interpolation uses hour/quarter centres and constant daily boundary extension. Raw interpolation violates hourly mean constraint, corrected version passes. Script: evaluate_baselines.py. No fitting or active jobs; these are training reconstruction diagnostics.
