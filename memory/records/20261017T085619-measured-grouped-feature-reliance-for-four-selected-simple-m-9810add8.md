---
title: Measured grouped feature reliance for four selected simple models
status: current
kind: analysis
recorded_at: '2026-10-17T08:56:19.004597+00:00'
sources:
- analyze_feature_importance.py
- results/feature_importance_20261017/ANALYSIS.md
- results/feature_importance_20261017/output/importance.json
- results/simple_5090/selection.json
experiment_id: null
supersedes: []
---

2026-10-17 user requested feature importance, specifically relevance of interpolation-error lags1/2/7. Ran 20 whole-day grouped permutations per group per month, August and September, frozen August-selected linear168/additive173/MLP190/CNN162. Ando short CPU analysis in /tmp/hackaton_feature_importance_20261017, completed; artifacts copied locally. Baseline replay assertions passed. September delta MAE all interpolation-error history: .4623/.3857/.1049/.0608; analog history .3697/.2845/.0746/.0370; exact lags1/2/7 combined .0300/.0479/.00016/.00451. Linear/additive strongly rely on aggregate/analog history; MLP slopes and hourly curves; CNN daily hourly curves1.0943. Exact lag1 clearest for linear/additive; lag2/7 weak. Seven-day input has not been trained. Detailed grouped results, descriptive circular7-day block bootstrap, method and caveats in ANALYSIS.md and output/importance.json. Whole-day permutations cannot assess constant hour/zone identity; calendar limited to within-month changes. Fixed anchor differs by family; correlated features and out-of-distribution perturbation prevent treating reliance as causal/retraining benefit. Covers four simple winners, not monthly blend/submission. No active analysis job remains. Next optional matched group-drop retraining.
