# RevIN quantile matched comparison

Eight developmental expanding folds. Compact CNN with fixed physical units versus input-day reversible normalization with learned positive affine parameters. Both arms fit from scratch with identical lr/seed budgets and original-currency loss. Lower mean monthly MAE/interpolation MAE is better.

| Recipe | Mean monthly ratio | Pooled MAE | Calibrated90 coverage | Width |
|---|---:|---:|---:|---:|
| compact_fixed | 0.768726 | 4.456372 | 89.70% | 20.311 |
| validation_selector | 0.772731 | 4.482381 | 90.66% | 21.511 |
| compact_day_reversible | 0.787509 | 4.574789 | 91.27% | 24.145 |
| interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |

This is RevIN adapted to constrained interpolation residuals, not a claim of exact published architecture replication. Learned affine input bias is not added to output deviations; interpolation restores the known hourly level. Scale remains positive and all quantiles use the same inverse affine. No deployment promotion. See UNCERTAINTY.md for original and normalized bands, interval coverage, paired April26 ZONE_3 and March14 ZONE_2 examples, and error associations.
