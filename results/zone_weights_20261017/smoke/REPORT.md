# Posthoc zone weight experiment

Frozen six-expert quantile predictions; GPU-fitted convex weights, separate interval calibration, chronological next-month tests. Developmental results, not a blind country-transfer test.

| Recipe | Mean monthly MAE/interpolation MAE | Pooled MAE | 90% coverage | 90% width |
|---|---:|---:|---:|---:|
| zone_selected_ratio | 0.758387 | 2.918949 | 92.35% | 15.563 |
| zone_fraction_0.5_ratio | 0.758490 | 2.919346 | 93.25% | 16.048 |
| common_selected_ratio | 0.758501 | 2.919391 | 93.01% | 15.940 |
| zone_fraction_0.75_ratio | 0.758501 | 2.919391 | 93.01% | 15.940 |
| zone_fraction_0.25_ratio | 0.758839 | 2.920689 | 93.41% | 16.156 |
| zone_ratio | 0.758909 | 2.920960 | 92.38% | 15.832 |
| global_ratio | 0.759659 | 2.923845 | 93.49% | 16.264 |
| original_blend_uniform | 0.762760 | 2.935782 | 93.19% | 16.024 |
| original_validation_selector | 0.762760 | 2.935782 | 93.19% | 16.024 |
| pretest_selector | 0.763298 | 2.937854 | 92.83% | 15.517 |
| zone_selected_uniform | 0.763298 | 2.937854 | 92.83% | 15.517 |
| zone_fraction_0.5_uniform | 0.764691 | 2.943212 | 92.98% | 15.735 |
| original_blend_ratio | 0.764705 | 2.943269 | 93.53% | 16.972 |
| zone_fraction_0.25_uniform | 0.764870 | 2.943903 | 93.02% | 15.682 |
| common_selected_uniform | 0.764930 | 2.944132 | 92.96% | 15.791 |
| zone_fraction_0.75_uniform | 0.764930 | 2.944132 | 92.96% | 15.791 |
| global_uniform | 0.765560 | 2.946556 | 93.05% | 15.653 |
| zone_uniform | 0.765567 | 2.946585 | 92.88% | 15.859 |
| original_interpolation | 1.000000 | 3.848893 | 94.60% | 23.408 |

Global uses six shared weights; zone uses six weights separately for each zone. Fractions shrink zone weights toward the global fit. Selected fractions use fit days1–10 and tuning days11–15, then refit on all15. The pretest selector selects objective and common/per-zone policy before the next test month. Expert checkpoint selection already saw all15 days, so the inner tuning score is not independently nested.

All quantile levels and all four quarters share a zone weight vector. The median preserves supplied hourly means; final intervals are calibrated separately on days16–end. Confidence bins summarize relative calibrated width, not correctness probabilities.

Independent first-fold linear-program check: maximum GPU fitting MAE excess 0.00000010 EUR/MWh. Full constraint/calibration/source replay is in audit.json.

Artifacts: monthly_scores.csv, monthly_zone_scores.csv, zone_summary.csv, weights.csv, interval_metrics.csv, confidence_bins.csv, predictions.csv.gz, folds/*/selected/*/predictions.npz. No production model was replaced.
