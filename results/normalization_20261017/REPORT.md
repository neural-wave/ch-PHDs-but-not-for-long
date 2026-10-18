# Matched normalization comparison

Eight developmental monthly backtests. Every arm has identical family-specific parameter counts, seeds, learning-rate grid, physical-currency loss and evaluation windows. Lower mean monthly MAE/interpolation MAE is better.

| Recipe | Mean monthly ratio | Pooled MAE | Calibrated 90% coverage | Width |
|---|---:|---:|---:|---:|
| s5_fixed | 0.762926 | 4.422713 | 89.37% | 19.401 |
| s4d_fixed | 0.763497 | 4.430358 | 89.70% | 19.954 |
| compact_fixed | 0.769238 | 4.458661 | 89.74% | 20.338 |
| validation_selector | 0.774028 | 4.490647 | 90.08% | 20.853 |
| s4d_standard | 0.782518 | 4.530352 | 88.67% | 19.960 |
| s5_standard | 0.789864 | 4.576548 | 89.12% | 20.627 |
| s4d_day_input | 0.796951 | 4.625565 | 89.87% | 21.501 |
| compact_standard | 0.797334 | 4.628212 | 89.18% | 21.071 |
| s5_day_input | 0.799327 | 4.640775 | 90.13% | 21.870 |
| s4d_day_reversible | 0.802287 | 4.662785 | 91.18% | 24.694 |
| s5_day_reversible | 0.805933 | 4.681590 | 89.95% | 22.809 |
| compact_day_input | 0.807165 | 4.687277 | 88.54% | 20.564 |
| compact_day_reversible | 0.812202 | 4.714042 | 90.11% | 23.496 |
| interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |

Paired differences below are arm minus standard; negative favors the arm. Each value averages eight months and the same four lr/seed combinations. No test result selects a new recipe.

| Family | Fixed minus standard | Day input minus standard | Day reversible minus standard |
|---|---:|---:|---:|
| compact | -0.031447 | +0.003822 | +0.019805 |
| s4d | -0.024915 | +0.020253 | +0.018522 |
| s5 | -0.026624 | +0.006282 | +0.013895 |

All candidate pairings: paired_candidate_ratios.csv. Independently validation-selected pairings: paired_selected_ratios.csv. Complete quantiles/CQR intervals: folds/<month>/selected/<recipe>/predictions.npz. Width/error diagnostics: confidence_bins.csv and fold metrics. Daily scaling uses only known hourly inputs, keeps absolute level/scale side features, and has no learned affine parameters. Fixed retains engineered physical units and clipping. Global z-scores are fit using fitting dates only. No deployment replacement.
