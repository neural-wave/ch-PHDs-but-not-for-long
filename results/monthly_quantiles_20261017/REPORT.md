# Monthly quantile comparison and uncertainty

Every learned recipe predicts eleven ordered quantiles. The constrained q50 conserves supplied hourly means and determines MAE/interpolation_MAE. Nonmedian marginal quantiles do not have an hourly-mean constraint. All test months remain developmental.

| Recipe | Mean monthly MAE ratio | Pooled MAE | Mean monthly pinball ratio | Calibrated90%coverage | Calibrated90%width |
|---|---:|---:|---:|---:|---:|
| blend_uniform | 0.720494 | 4.176488 | 0.211631 | 92.30% | 20.7515 |
| validation_selector | 0.721285 | 4.181172 | 0.212818 | 92.58% | 21.4882 |
| blend_ratio | 0.721680 | 4.182921 | 0.213324 | 92.73% | 21.9636 |
| linear_compact_ratio | 0.730386 | 4.234448 | 0.220638 | 92.13% | 22.8672 |
| linear_compact_uniform | 0.730864 | 4.237331 | 0.219129 | 91.39% | 21.3375 |
| linear_rich_ratio | 0.734666 | 4.253466 | 0.234517 | 95.75% | 30.8073 |
| cnn2d_analog_ratio | 0.735008 | 4.258815 | 0.226059 | 89.89% | 21.3940 |
| linear_rich_uniform | 0.735547 | 4.260377 | 0.227393 | 94.85% | 27.2789 |
| cnn2d_analog_uniform | 0.736778 | 4.269359 | 0.226647 | 89.99% | 21.2694 |
| additive_compact_uniform | 0.737004 | 4.275943 | 0.216428 | 90.32% | 20.3392 |
| additive_compact_ratio | 0.737317 | 4.277164 | 0.216920 | 90.56% | 20.6076 |
| mlp_compact_uniform | 0.746533 | 4.333533 | 0.222808 | 89.97% | 19.8299 |
| mlp_compact_ratio | 0.750620 | 4.361057 | 0.223329 | 90.29% | 20.0431 |
| cnn_rich_ratio | 0.765939 | 4.440318 | 0.230756 | 89.58% | 20.9014 |
| cnn_rich_uniform | 0.768762 | 4.454864 | 0.231297 | 89.23% | 20.2826 |
| interpolation | 1.000000 | 5.814817 | 0.309732 | 89.50% | 27.7836 |

Each fold selects/stops on the preceding month's first15days; remaining days calibrate intervals. No outer-test errors select a candidate, blend, calibration correction or confidence cutoff.

Use predictions.csv.gz to inspect the validation-selected recipe quarter by quarter. It includes all eleven quantiles, calibrated90%bounds/width and calibration-relative uncertainty percentile. Narrow/middle/wide thresholds are0.25/0.75. These labels describe relative interval width, not a probability that an individual prediction is accurate. Every model's full raw quantiles and calibrated50/80/90/95%bounds are also in folds/<month>/selected/<recipe>/predictions.npz.

Inspect confidence_bins.csv for whether narrow predictions actually had smaller errors, their same-bin interpolation baseline and empirical90%coverage. interval_metrics.csv includes raw/calibrated50/80/90/95%coverage/width plus whole-hour/day coverage. Pointwise calibration is not a simultaneous-day guarantee; time dependence and drift require empirical assessment.

The validation_selector is chosen from preceding-block performance; the lowest aggregate outer-test recipe is descriptive research evidence. Quantile blends are convex averages of quantile functions, not quantiles of a mixture distribution. Their calibration is fitted separately. Earlier point-only results are preserved as historical controls. No catastrophe classifier or deployment replacement was made.

Reproduction and fixed budget: campaigns/monthly_quantiles_20261017/PROTOCOL.md and run.py. Source hashes/checkpoints/predictions/selections/calibration receipts and the independent audit are retained.
