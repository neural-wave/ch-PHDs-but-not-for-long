# Strong six-expert ensemble: normalization comparison

Eight developmental tests; original ordinary-training standardized controls reused exactly,192 new fixed-scaling fits. Lower mean monthly MAE/interpolation MAE is better. Pure standardized blend must reproduce .7204941118854624.

| Recipe | Mean monthly ratio | Pooled MAE | Calibrated90% coverage | Width |
|---|---:|---:|---:|---:|
| blend__mixed | 0.708371 | 4.106785 | 90.35% | 18.342 |
| validation_selector | 0.708371 | 4.106785 | 90.35% | 18.342 |
| blend__fixed | 0.710359 | 4.118345 | 90.02% | 18.162 |
| blend__standard | 0.720494 | 4.176488 | 92.30% | 20.752 |
| linear_compact__fixed | 0.725707 | 4.207480 | 89.10% | 18.471 |
| cnn2d_analog__fixed | 0.727906 | 4.224506 | 90.52% | 21.387 |
| linear_rich__fixed | 0.728020 | 4.217723 | 89.86% | 19.379 |
| additive_compact__fixed | 0.730142 | 4.233447 | 91.28% | 21.129 |
| linear_compact__standard | 0.730864 | 4.237331 | 91.39% | 21.337 |
| mlp_compact__fixed | 0.734969 | 4.266283 | 90.08% | 18.874 |
| linear_rich__standard | 0.735547 | 4.260377 | 94.85% | 27.279 |
| cnn2d_analog__standard | 0.736778 | 4.269359 | 89.99% | 21.269 |
| additive_compact__standard | 0.737004 | 4.275943 | 90.32% | 20.339 |
| cnn_rich__fixed | 0.738978 | 4.286494 | 87.93% | 17.880 |
| mlp_compact__standard | 0.746533 | 4.333533 | 89.97% | 19.830 |
| cnn_rich__standard | 0.768762 | 4.454864 | 89.23% | 20.283 |
| interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |

Fixed retains engineered physical units and clip; standard adds fitting-only z-scores. All models and recipes predict eleven ordered quantiles. Family hyperparameters, blend menus and mixed per-family arm choices use preceding first15days only; CQR uses16–end. No outer-test selection or deployment update. The previous compact-summary CNN/S4D/S5 comparison used different models and cannot determine this ensemble result.

All candidate pairings: paired_candidate_ratios.csv. Full quantiles/intervals and width/error diagnostics retained. The .718907 zone-shrinkage result belongs to a separate posthoc study and is not the baseline altered here.
