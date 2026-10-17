# One-layer linear S4D frequency study

Developmental backtests; full VALIDATION_PROTOCOL.md windows, fitting-only preprocessing, frozen test parameters. Primary score is arithmetic mean of eight monthly ratios.

| Recipe | Mean monthly ratio | Pooled MAE | Calibrated 90% coverage | Width |
|---|---:|---:|---:|---:|
| learned_frequencies | 0.810498 | 4.698031 | 90.79% | 24.309 |
| validation_selector | 0.810498 | 4.698031 | 90.79% | 24.309 |
| fixed_frequencies | 0.817796 | 4.741722 | 91.27% | 25.699 |
| interpolation | 1.000000 | 5.814817 | 89.50% | 27.784 |

Point path: one shared bidirectional linear S4D and a linear readout, followed by hourly mean projection. No nonlinear features, activation, LayerNorm or clipping. Quantile gaps use a separate softplus head. Exact source-channel contributions and independent convolution replay passed for all16 family/fold winners. Frequencies and decay times are learned filter modes, not uniquely identified physical cycles; inspect effective kernels and cross-fold consistency.
