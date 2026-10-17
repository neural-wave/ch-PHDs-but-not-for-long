# Expanding-training optimization with fixed physical feature scaling

Every learned recipe predicts eleven ordered quantiles. The constrained q50 conserves supplied hourly means and determines MAE/interpolation_MAE. Nonmedian marginal quantiles do not have an hourly-mean constraint. All test months remain developmental.

| Recipe | Mean monthly MAE ratio | Pooled MAE | Mean monthly pinball ratio | Calibrated90%coverage | Calibrated90%width |
|---|---:|---:|---:|---:|---:|
| fixed_blend_ratio | 0.707968 | 4.103184 | 0.207375 | 90.59% | 18.8354 |
| fixed_blend_uniform | 0.708717 | 4.108689 | 0.207156 | 90.42% | 18.5942 |
| combined_blend_uniform | 0.708887 | 4.107368 | 0.207832 | 90.62% | 18.7849 |
| validation_selector | 0.708892 | 4.107397 | 0.207926 | 90.71% | 18.8409 |
| combined_blend_ratio | 0.709132 | 4.110525 | 0.207648 | 90.69% | 18.9999 |
| standard_blend_uniform | 0.720494 | 4.176488 | 0.211631 | 92.30% | 20.7515 |
| standard_blend_ratio | 0.721680 | 4.182921 | 0.213324 | 92.73% | 21.9636 |
| fixed_cnn2d_analog_uniform | 0.722965 | 4.190350 | 0.219355 | 90.67% | 21.0983 |
| fixed_cnn2d_analog_ratio | 0.724033 | 4.198922 | 0.221595 | 90.55% | 21.2530 |
| fixed_linear_compact_ratio | 0.724553 | 4.201780 | 0.215858 | 89.01% | 18.2373 |
| fixed_linear_compact_uniform | 0.725535 | 4.206535 | 0.216910 | 89.06% | 18.3917 |
| fixed_linear_rich_ratio | 0.726839 | 4.211747 | 0.215294 | 89.72% | 19.2439 |
| fixed_mlp_compact_uniform | 0.726885 | 4.218104 | 0.211060 | 89.97% | 18.5616 |
| fixed_cnn_rich_uniform | 0.727913 | 4.220295 | 0.218273 | 88.55% | 17.8246 |
| fixed_additive_compact_ratio | 0.728128 | 4.222913 | 0.213261 | 91.51% | 21.3297 |
| fixed_linear_rich_uniform | 0.728609 | 4.220166 | 0.216106 | 89.79% | 19.2650 |
| fixed_additive_compact_uniform | 0.729586 | 4.230943 | 0.213361 | 91.29% | 20.9662 |
| fixed_mlp_compact_ratio | 0.730118 | 4.239394 | 0.211968 | 89.83% | 18.5864 |
| standard_linear_compact_ratio | 0.730386 | 4.234448 | 0.220638 | 92.13% | 22.8672 |
| standard_linear_compact_uniform | 0.730864 | 4.237331 | 0.219129 | 91.39% | 21.3375 |
| fixed_cnn_rich_ratio | 0.732949 | 4.249768 | 0.223100 | 88.53% | 19.1860 |
| standard_linear_rich_ratio | 0.734666 | 4.253466 | 0.234517 | 95.75% | 30.8073 |
| standard_cnn2d_analog_ratio | 0.735008 | 4.258815 | 0.226059 | 89.89% | 21.3940 |
| standard_linear_rich_uniform | 0.735547 | 4.260377 | 0.227393 | 94.85% | 27.2789 |
| standard_cnn2d_analog_uniform | 0.736778 | 4.269359 | 0.226647 | 89.99% | 21.2694 |
| standard_additive_compact_uniform | 0.737004 | 4.275943 | 0.216428 | 90.32% | 20.3392 |
| standard_additive_compact_ratio | 0.737317 | 4.277164 | 0.216920 | 90.56% | 20.6076 |
| standard_mlp_compact_uniform | 0.746533 | 4.333533 | 0.222808 | 89.97% | 19.8299 |
| standard_mlp_compact_ratio | 0.750620 | 4.361057 | 0.223329 | 90.29% | 20.0431 |
| standard_cnn_rich_ratio | 0.765939 | 4.440318 | 0.230756 | 89.58% | 20.9014 |
| standard_cnn_rich_uniform | 0.768762 | 4.454864 | 0.231297 | 89.23% | 20.2826 |
| interpolation | 1.000000 | 5.814817 | 0.309732 | 89.50% | 27.7836 |

Each fold selects/stops on the preceding month's first15days; remaining days calibrate intervals. No outer-test errors select a candidate, blend, calibration correction or confidence cutoff.

Use predictions.csv.gz to inspect the validation-selected recipe quarter by quarter. It includes all eleven quantiles, calibrated90%bounds/width and calibration-relative uncertainty percentile. Narrow/middle/wide thresholds are0.25/0.75. These labels describe relative interval width, not a probability that an individual prediction is accurate. Every model's full raw quantiles and calibrated50/80/90/95%bounds are also in folds/<month>/selected/<recipe>/predictions.npz.

Inspect confidence_bins.csv for whether narrow predictions actually had smaller errors, their same-bin interpolation baseline and empirical90%coverage. interval_metrics.csv includes raw/calibrated50/80/90/95%coverage/width plus whole-hour/day coverage. Pointwise calibration is not a simultaneous-day guarantee; time dependence and drift require empirical assessment.

The validation_selector is chosen from preceding-block performance; the lowest aggregate outer-test recipe is descriptive research evidence. Quantile blends are convex averages of quantile functions, not quantiles of a mixture distribution. Their calibration is fitted separately. Earlier point-only results are preserved as historical controls. No catastrophe classifier or deployment replacement was made.

Reproduction and fixed budget: campaigns/minimize_expanding_20261017/PROTOCOL.md and run.py. Source hashes/checkpoints/predictions/selections/calibration receipts and the independent audit are retained.

Training expands from October–December2025 to October2025–July2026. No training-window shortening or selection-date refit.

Original pre-test selector: 0.721285; new pre-test selector: 0.708892; relative MAE-ratio change: -1.718%.
Original ordinary blend observed reference: 0.720494. The new aggregate winner is descriptive and is not substituted for the pre-test selector.

New fits use fixed engineered physical units and clip15. Frozen standard controls are reused from the audited prior campaign; architecture, loss, learning rates and seeds match. Family winners can average the two seeds at a common learning rate. Blends are global convex quantile-function averages, selected on days1–15 and calibrated on16–end.
