---
title: User prioritizes intrinsic interpretability and difficult-day robustness
status: current
kind: decision
recorded_at: '2026-10-17T09:08:52.826673+00:00'
sources:
- INTERPRETABLE_ROBUSTNESS_PLAN.md
- INTERPRETABLE_PROTOCOL.md
- results/feature_importance_20261017/ANALYSIS.md
experiment_id: null
supersedes: []
---

2026-10-17 user explicitly liked compact confidence-weighted historical-analog residual models and requested focus on interesting interpretable models and robustness, including especially bad days. Saved concrete design in INTERPRETABLE_ROBUSTNESS_PLAN.md: compact example-based analog residuals, MAE-trained sparse additive contrast corrections, and tiny observable rule gate. Proposed difficult-day diagnostic must distinguish detectable high error from predictable relative benefit of fallback. Fit gates only on chronological out-of-fold errors and permitted inputs. Shared within-hour shrinkage preserves means; quarterwise weighted median residuals need projection and final MAE evaluation. Evaluate official average MAE alongside fixed common difficult-day subsets and tail/worst-month diagnostics. Interpolation is not assumed safe; tail optimization may trade average MAE. Existing compact interpretable benchmark uses squared-error ridge, whereas new proposal optimizes reconstructed quarter MAE. No experiment launched or improved score claimed; previous remote job state not rechecked.
