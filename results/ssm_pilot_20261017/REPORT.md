# S4D/S5 history pilot — audited results

September developmental; earlier fixed-config folds also historical development, not a new blind test.

| Family | Parameters | August MAE | September MAE | Earlier rolling MAE |
|---|---:|---:|---:|---:|
| s4d | 25740 | 4.558687 | 5.497365 | 4.357039 |
| s5 | 25756 | 4.552202 | 5.603856 | 4.391647 |
| gru | 29964 | 4.567998 | 5.517535 | 4.356461 |
| tcn | 51564 | 4.591878 | 5.452077 | 4.386337 |
| compact | 13900 | 4.562378 | 5.591747 | 4.359954 |

Overall August-selected trial: 14. Interpolation September MAE: 7.090321.

Earlier rolling models use fixed width32/lr.001/seed17, not August-selected hyperparameters. All models share current compact causal summaries; compact control omits only the sequential history encoder. TCN has more parameters; report counts rather than claim parameter parity.

Maximum hourly mean error: 1.14e-13. All56 checkpoints replayed. No deployment changed.
