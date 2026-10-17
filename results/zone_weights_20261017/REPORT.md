# Posthoc zone weight experiment

Frozen six-expert quantile predictions; GPU-fitted convex weights, separate interval calibration, chronological next-month tests. Developmental results, not a blind country-transfer test.

| Recipe | Mean monthly MAE/interpolation MAE | Pooled MAE | 90% coverage | 90% width |
|---|---:|---:|---:|---:|
| zone_fraction_0.25_uniform | 0.718907 | 4.165801 | 91.66% | 20.019 |
| zone_fraction_0.5_uniform | 0.718985 | 4.166422 | 91.64% | 19.989 |
| common_selected_uniform | 0.719180 | 4.167354 | 91.63% | 20.011 |
| zone_selected_uniform | 0.719310 | 4.168573 | 91.52% | 19.968 |
| global_uniform | 0.719311 | 4.167925 | 91.70% | 20.087 |
| zone_selected_ratio | 0.719338 | 4.171249 | 92.08% | 20.844 |
| zone_fraction_0.5_ratio | 0.719453 | 4.171662 | 92.20% | 20.978 |
| zone_fraction_0.75_uniform | 0.719504 | 4.169568 | 91.49% | 19.943 |
| zone_fraction_0.25_ratio | 0.719519 | 4.172039 | 92.28% | 21.039 |
| common_selected_ratio | 0.719632 | 4.172963 | 92.14% | 20.890 |
| zone_fraction_0.75_ratio | 0.719828 | 4.173853 | 92.13% | 20.965 |
| pretest_selector | 0.719952 | 4.173446 | 92.14% | 20.838 |
| global_ratio | 0.720002 | 4.174800 | 92.32% | 21.098 |
| zone_uniform | 0.720485 | 4.175361 | 91.33% | 19.922 |
| original_blend_uniform | 0.720494 | 4.176488 | 92.30% | 20.752 |
| zone_ratio | 0.720644 | 4.178603 | 92.02% | 20.983 |
| original_validation_selector | 0.721285 | 4.181172 | 92.58% | 21.488 |
| original_blend_ratio | 0.721680 | 4.182921 | 92.73% | 21.964 |
| original_interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |

Global uses six shared weights; zone uses six weights separately for each zone. Fractions shrink zone weights toward the global fit. Selected fractions use fit days1–10 and tuning days11–15, then refit on all15. The pretest selector selects objective and common/per-zone policy before the next test month. Expert checkpoint selection already saw all15 days, so the inner tuning score is not independently nested.

All quantile levels and all four quarters share a zone weight vector. The median preserves supplied hourly means; final intervals are calibrated separately on days16–end. Confidence bins summarize relative calibrated width, not correctness probabilities.

Independent first-fold linear-program check: maximum GPU fitting MAE excess 0.00000010 EUR/MWh. Full constraint/calibration/source replay is in audit.json.

Artifacts: monthly_scores.csv, monthly_zone_scores.csv, zone_summary.csv, weights.csv, interval_metrics.csv, confidence_bins.csv, predictions.csv.gz, folds/*/selected/*/predictions.npz. No production model was replaced.
