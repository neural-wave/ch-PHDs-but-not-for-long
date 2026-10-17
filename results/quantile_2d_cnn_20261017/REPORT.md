# Small 2D CNN with quantiles and day-error risk

Selection: August1–15. Calibration: August16–31. September is developmental.

August-selected point model: **analog90_0**. Quantile model: **analog90_1**.

| Input/loss | Point parameters | Total parameters | Selection MAE | September MAE |
|---|---:|---:|---:|---:|
| raw_0 | 3860 | 4608 | 4.735579 | 5.462844 |
| raw_1 | 3860 | 4608 | 4.733655 | 5.427657 |
| analog90_0 | 4948 | 5696 | 4.509062 | 5.336198 |
| analog90_1 | 4948 | 5696 | 4.519095 | 5.216651 |

The practical target is approximately5.2 MAE; no September-based search or stopping. Only point predictions conserve hourly means.

| Quantile model | Raw90% coverage | Calibrated90% coverage | Calibrated width |
|---|---:|---:|---:|
| raw_1 | 0.896 | 0.909 | 28.780 |
| analog90_1 | 0.912 | 0.924 | 28.993 |

Severe day: joint four-zone day MAE > 10.0 EUR/MWh. Risk labels: 182 held-out earlier days, 1 events.

| Risk features | September events | Brier | Average precision | Recall among top10% |
|---|---:|---:|---:|---:|
| inputs | 0 | 0.000 | n/a | n/a |
| inputs_uncertainty | 0 | 0.000 | n/a | n/a |
| plus_conv | 0 | 0.000 | n/a | n/a |
| plus_fusion | 0 | 0.000 | n/a | n/a |

Risk fit on chronological out-of-sample errors; activation summaries are channel-permutation invariant, not neuron identities. Calibration has16days only; probabilities exploratory. All probes reported without September selection. Top10% recall is a batch ranking diagnostic; deployable threshold fixed on calibration.

Calibration uses August16-31 only. Pointwise empirical coverage under temporal dependence; no exchangeability or whole-day guarantee.

See explanations/ for normalized kernels, signed maps, randomization checks, and consistent history ablations.

Independent artifact audit: **PASSED**.

Exploratory continuous daily-error probes (the primary severe-day test remains unchanged):

| Features | Spearman with daily MAE | Error predicting daily MAE |
|---|---:|---:|
| inputs | 0.410 | 0.846 |
| inputs_uncertainty | 0.364 | 0.789 |
| plus_conv | 0.193 | 0.958 |
| plus_fusion | 0.248 | 0.888 |

Constant prediction from calibration days: error 1.009 EUR/MWh.
Raw90% interval width versus actual daily MAE: Spearman 0.304.

Exploratory continuous-error diagnostic added after the predefined10EUR/MWh severe-day test had only1 training event and0 September events. All four fixed probes reported; no model selection or reconstruction changes. Fit182 earlier held-out errors, offset on August16-31 only.30 Septemberdays cannot establish reliable catastrophe detection.

Independent artifact audit: **PASSED**.
