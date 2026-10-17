# Eight-fold S4D/S5 quantile comparison

Developmental backtests; full VALIDATION_PROTOCOL.md windows, fitting-only preprocessing, frozen test parameters. Primary score is arithmetic mean of eight monthly ratios.

| Recipe | Mean monthly ratio | Pooled MAE | Calibrated 90% coverage | Width |
|---|---:|---:|---:|---:|
| tcn | 0.778942 | 4.512701 | 90.16% | 20.625 |
| s4d | 0.785554 | 4.551869 | 88.62% | 20.029 |
| gru | 0.785659 | 4.551947 | 89.08% | 20.276 |
| validation_selector | 0.787638 | 4.566457 | 90.31% | 21.515 |
| s5 | 0.788695 | 4.568278 | 89.23% | 20.625 |
| compact | 0.797603 | 4.629994 | 89.17% | 21.067 |
| interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |
