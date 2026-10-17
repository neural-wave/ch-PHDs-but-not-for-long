---
title: 'Winning-direction recommendation: historical analogs, robust validation and
  justified complexity'
status: current
kind: strategy
recorded_at: '2026-10-17T08:59:43.391576+00:00'
sources:
- results/challenge_rules_extracted.txt
- results/feature_importance_20261017/ANALYSIS.md
- results/simple_5090/selection.json
- results/monthly_joint_blend_gpu1/metrics.json
experiment_id: null
supersedes: []
---

2026-10-17: User asked which direction maximizes chances of winning given noisy targets and many models near MAE5. Recommendation, not an authorized new experiment or established improvement: prioritize reliable temporal generalization and conditional historical-analog residual corrections over further broad architecture HPO. Scores near5 do not establish an irreducible floor. Verified September development scores: linear5.209794, CNN5.222494, MLP5.343947, monthly joint blend5.091209; August picked CNN, demonstrating month-dependent rankings. Rules lines245–305 explicitly make MAE only one component of final jury evaluation and assess justified complexity, robustness, clarity, interpretability and reproducibility; no numeric qualitative weighting specified. Frozen-model permutation evidence supports analog interpolation-error history for linear/additive, but is neither retraining evidence nor tested monthly-blend reliance. Proposed focused study: tune causal analog matching/recency and slope, cross-zone spread, signed residual and volatility inputs; use small regularized correction plus confidence shrinkage, jointly train quarter MAE under exact hourly-mean constraints. Compare few candidates using nested chronological selection, paired day/month gains and temporal block uncertainty; existing data including September remains development evidence. Analyze error contribution by observable regime and zone; zone2 is hardest but all zones carry equal official weight. Small convex ensemble/distillation only if consistent out-of-fold gains justify complexity. Conformal intervals alone do not improve point MAE. No remote state checked and no jobs launched or changed in this strategy turn.
