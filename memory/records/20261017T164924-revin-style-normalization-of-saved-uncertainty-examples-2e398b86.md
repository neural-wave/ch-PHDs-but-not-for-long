---
title: RevIN-style normalization of saved uncertainty examples
status: current
kind: experiment
recorded_at: '2026-10-17T16:49:24.752061+00:00'
sources:
- results/quantile_examples_20261017/REPORT.md
- campaigns/normalization_20261017/PROTOCOL.md
- results/normalization_20261017/REPORT.md
experiment_id: null
supersedes: []
---

User asked to try RevIN after discussing normalized uncertainty plots. Asked clarification display vs retraining; independently completed display-only transform of existing champion examples. Known hourly day/zone mu and sqrt(var+1), same statistics for observations/quantiles/bounds/interpolation, shared y axis. No new training/inference or learned affine. Width0.49803 vs0.78709, normalized MAE0.23197 vs0.07154; wider drop bands correspond to lower normalized error in these selected extrema. Affine round-trip verified and image inspected. Existing normalization campaign already has eight-fold day_reversible quantile controls (without learned affine), overall worse than fixed controls; no exact RevIN/new drop-specific fitted comparison claimed. Artifacts results/quantile_examples_20261017/revin_coordinates.png/svg/json; report updated; code/plot_revin_examples.py. Pending user clarification if actual retraining intended.
