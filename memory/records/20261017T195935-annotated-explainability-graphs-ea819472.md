---
title: Annotated explainability graphs
status: current
kind: experiment
recorded_at: '2026-10-17T19:59:35.317798+00:00'
sources:
- results/interpretability_deep_20261017/REPORT.md
- campaigns/interpretability_explained_20261017/plot.py
experiment_id: null
supersedes: []
---

Created four visually inspected PNGs in results/interpretability_explained_20261017/ using campaigns/interpretability_explained_20261017/plot.py. Small local Pillow rendering of saved results/interpretability_deep_20261017/ JSONs; no fitting or inference. Graphs explain linear feature contributions, selected success/failure cases, equal-month champion gains by curve and uncertainty, and June 1 ensemble cancellation. Feature contributions cover one September checkpoint and are not causal or ablation gains; examples are hindsight-selected extremes; champion aggregates cover eight developmental folds. No model or submission changes.
