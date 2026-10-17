---
title: Linear and ensemble interpretation from saved coefficients and conditional
  metrics
status: current
kind: experiment
recorded_at: '2026-10-17T16:38:50.770567+00:00'
sources:
- results/interpretability_20261017/REPORT.md
- results/interpretability_20261017/summary.json
- campaigns/interpretability_20261017/analyze.py
experiment_id: null
supersedes: []
---

Applied metadata-only interpretation on 2026-10-17; no training/inference/model changes. Reproducible campaign script exports all 389 standalone fixed-September linear quarter coefficient effects, verifies zero-sum decomposition, and summarizes current champion eight-month zone/uncertainty scores. Report clearly separates standalone checkpoint coefficients, current champion composition/conditional metrics, and earlier ensemble specialization/cancellation evidence. Largest linear sensitivities include analogous-history residuals; coefficients are conditional sensitivities, not causal or held-out importance. Champion zone ratios .591937/.652059/.822717/.783267; equal-month narrow/middle/wide MAE 2.026/3.508/6.556. Further per-example/fold stability and perturbation replay not executed; substantial replay requires research server.
