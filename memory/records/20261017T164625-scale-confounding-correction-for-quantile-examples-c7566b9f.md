---
title: Scale confounding correction for quantile examples
status: current
kind: experiment
recorded_at: '2026-10-17T16:46:25.885767+00:00'
sources:
- results/quantile_examples_20261017/REPORT.md
- results/quantile_examples_20261017/scale_diagnostic.json
experiment_id: null
supersedes: []
---

User flagged raw uncertainty-width comparisons as scale-confounded. Recomputed fixed mean-corrected interpolation for saved two examples: March14 ZONE_2 model/interpolation MAE1.09231; April26 ZONE_3 ratio1.03130. Both underperform baseline. Width/hourly std0.49928 vs0.78711: raw15.66x contrast becomes1.58x. This illustrative scaling is not a validated uncertainty normalization; realized interpolation error is retrospective, not an available inference input. Raw pooled narrow/wide MAE does not establish predictive uncertainty beyond scale. See results/quantile_examples_20261017/scale_diagnostic.json and updated REPORT.md. No inference/training.
