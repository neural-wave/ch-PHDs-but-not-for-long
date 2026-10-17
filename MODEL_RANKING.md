# Model ranking — 17 October 2026

Standing rule: every future predictive comparison and research promotion uses [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md). Eight expanding February–September tests, preceding-month days 1–15 selection/16–end calibration, fresh fitting, and equal-month MAE/mean-corrected-interpolation MAE. Lower is better.

**Current research champion:** `minimize_expanding_20261017/fixed_blend_ratio` — **0.707968**, **29.20% less MAE than interpolation**, pooled MAE **4.103184 EUR/MWh**.

This user-authorized promotion is the best observed developmental recipe. It does not claim blind-test generalization or statistical significance. The strongest per-fold prior-data selection policy is separately identified in [best_model.json](best_model.json).

Coverage: **142** reported eight-fold rows, **110** rows after declared reused-control aliases, **121** historical recipe/scope rows and **5197** locally available individual candidate records. The catalogue can repeat a trial across campaign and per-trial sources. Raw HPO trials without matched eight-fold evaluations receive no primary rank. Search budgets and inner objectives differ; this ranks complete saved recipes.

## Leading eight-fold recipes

| Rank | Model recipe | Mean monthly ratio | Reduction | Pooled MAE | Output |
|---:|---|---:|---:|---:|---|
| 1 | Fixed-scale ensemble with seed averaging · fixed_blend_ratio | 0.707968 | 29.20% | 4.103184 | 11 quantiles |
| 2 | Controlled six-expert normalization · blend__mixed | 0.708371 | 29.16% | 4.106785 | 11 quantiles |
| 2 | Controlled six-expert normalization · validation_selector | 0.708371 | 29.16% | 4.106785 | 11 quantiles |
| 4 | Fixed-scale ensemble with seed averaging · fixed_blend_uniform | 0.708717 | 29.13% | 4.108689 | 11 quantiles |
| 5 | Fixed-scale ensemble with seed averaging · combined_blend_uniform | 0.708887 | 29.11% | 4.107368 | 11 quantiles |
| 6 | Fixed-scale ensemble with seed averaging · validation_selector | 0.708892 | 29.11% | 4.107397 | 11 quantiles |
| 7 | Fixed-scale ensemble with seed averaging · combined_blend_ratio | 0.709132 | 29.09% | 4.110525 | 11 quantiles |
| 8 | Controlled six-expert normalization · blend__fixed | 0.710359 | 28.96% | 4.118345 | 11 quantiles |
| 9 | Quantile zone-weight study · zone_fraction_0.25_uniform | 0.718907 | 28.11% | 4.165801 | 11 quantiles |
| 10 | Quantile zone-weight study · zone_fraction_0.5_uniform | 0.718985 | 28.10% | 4.166422 | 11 quantiles |
| 11 | Quantile zone-weight study · common_selected_uniform | 0.719180 | 28.08% | 4.167354 | 11 quantiles |
| 12 | Quantile zone-weight study · zone_selected_uniform | 0.719310 | 28.07% | 4.168573 | 11 quantiles |
| 13 | Quantile zone-weight study · global_uniform | 0.719311 | 28.07% | 4.167925 | 11 quantiles |
| 14 | Quantile zone-weight study · zone_selected_ratio | 0.719338 | 28.07% | 4.171249 | 11 quantiles |
| 15 | Quantile zone-weight study · zone_fraction_0.5_ratio | 0.719453 | 28.05% | 4.171662 | 11 quantiles |
| 16 | Quantile zone-weight study · zone_fraction_0.75_uniform | 0.719504 | 28.05% | 4.169568 | 11 quantiles |
| 17 | Quantile zone-weight study · zone_fraction_0.25_ratio | 0.719519 | 28.05% | 4.172039 | 11 quantiles |
| 18 | Quantile zone-weight study · common_selected_ratio | 0.719632 | 28.04% | 4.172963 | 11 quantiles |
| 19 | Quantile zone-weight study · zone_fraction_0.75_ratio | 0.719828 | 28.02% | 4.173853 | 11 quantiles |
| 20 | Quantile zone-weight study · pretest_selector | 0.719952 | 28.00% | 4.173446 | 11 quantiles |
| 21 | Quantile zone-weight study · global_ratio | 0.720002 | 28.00% | 4.174800 | 11 quantiles |
| 22 | Point-model monthly-ratio search · blend_ratio | 0.720436 | 27.96% | 4.177390 | Historical point control |
| 23 | Quantile zone-weight study · zone_uniform | 0.720485 | 27.95% | 4.175361 | 11 quantiles |
| 24 | Original quantile ensemble · blend_uniform | 0.720494 | 27.95% | 4.176488 | 11 quantiles |
| 25 | Quantile zone-weight study · zone_ratio | 0.720644 | 27.94% | 4.178603 | 11 quantiles |

## Best result per model family

Descriptive winners within each family; normalization, features, loss and ensemble search spaces differ.

| Family | Best observed recipe | Mean monthly ratio | Pooled MAE |
|---|---|---:|---:|
| Ensemble | `minimize_expanding_20261017/fixed_blend_ratio` | 0.707968 | 4.103184 |
| Selection policy | `ensemble_normalization_20261017/validation_selector` | 0.708371 | 4.106785 |
| Analog 2D CNN | `minimize_expanding_20261017/fixed_cnn2d_analog_uniform` | 0.722965 | 4.190350 |
| Linear model | `minimize_expanding_20261017/fixed_linear_compact_ratio` | 0.724553 | 4.201780 |
| MLP | `minimize_expanding_20261017/fixed_mlp_compact_uniform` | 0.726885 | 4.218104 |
| Temporal CNN | `minimize_expanding_20261017/fixed_cnn_rich_uniform` | 0.727913 | 4.220295 |
| Additive model | `minimize_expanding_20261017/fixed_additive_compact_ratio` | 0.728128 | 4.222913 |
| Causal analog rule | `monthly_ratio_20261017/analog_shrink_fixed` | 0.739302 | 4.286808 |
| Linear quantile model | `normalized_validation_20261017/linear_quantile_median` | 0.739333 | 4.278685 |
| Ridge regression | `monthly_ratio_20261017/ridge_compact_uniform` | 0.746022 | 4.326771 |
| Hierarchical model | `normalized_validation_20261017/hierarchical` | 0.762695 | 4.419719 |
| S5 | `normalization_20261017/s5_fixed` | 0.762926 | 4.422713 |
| S4D | `normalization_20261017/s4d_fixed` | 0.763497 | 4.430358 |
| Direct neural model | `normalized_validation_20261017/direct` | 0.765104 | 4.433215 |
| GRU | `ssm_monthly_20261017/gru` | 0.785659 | 4.551947 |
| Riccardo zonal CNN | `riccardo_comparison_20261017/riccardo_reduced` | 0.796834 | 4.615017 |
| Affine S4D | `linear_s4d_20261017/learned_frequencies` | 0.810498 | 4.698031 |
| Corrected interpolation | `normalized_validation_20261017/interpolation` | 1.000000 | 5.814817 |
| Hourly-repeat baseline | `deterministic/hourly_repeat` | 1.412642 | 8.253379 |

## Complete primary ranking

All rows below have the same eight months/241 days/92,544 targets and a passed owning audit. Declared reused controls are retained in all_reported_models.csv with alias targets, rather than presented as independent improvements.

| Rank | Recipe ID | Mean monthly ratio | Pooled MAE | Worst month | Evidence |
|---:|---|---:|---:|---:|---|
| 1 | `minimize_expanding_20261017/fixed_blend_ratio` | 0.707968 | 4.103184 | 0.754293 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 2 | `ensemble_normalization_20261017/blend__mixed` | 0.708371 | 4.106785 | 0.752704 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 2 | `ensemble_normalization_20261017/validation_selector` | 0.708371 | 4.106785 | 0.752704 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 4 | `minimize_expanding_20261017/fixed_blend_uniform` | 0.708717 | 4.108689 | 0.751644 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 5 | `minimize_expanding_20261017/combined_blend_uniform` | 0.708887 | 4.107368 | 0.759890 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 6 | `minimize_expanding_20261017/validation_selector` | 0.708892 | 4.107397 | 0.759890 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 7 | `minimize_expanding_20261017/combined_blend_ratio` | 0.709132 | 4.110525 | 0.754293 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 8 | `ensemble_normalization_20261017/blend__fixed` | 0.710359 | 4.118345 | 0.752704 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 9 | `zone_weights_20261017/zone_fraction_0.25_uniform` | 0.718907 | 4.165801 | 0.774767 | [scores](results/zone_weights_20261017/summary.csv) |
| 10 | `zone_weights_20261017/zone_fraction_0.5_uniform` | 0.718985 | 4.166422 | 0.774006 | [scores](results/zone_weights_20261017/summary.csv) |
| 11 | `zone_weights_20261017/common_selected_uniform` | 0.719180 | 4.167354 | 0.775800 | [scores](results/zone_weights_20261017/summary.csv) |
| 12 | `zone_weights_20261017/zone_selected_uniform` | 0.719310 | 4.168573 | 0.777283 | [scores](results/zone_weights_20261017/summary.csv) |
| 13 | `zone_weights_20261017/global_uniform` | 0.719311 | 4.167925 | 0.775800 | [scores](results/zone_weights_20261017/summary.csv) |
| 14 | `zone_weights_20261017/zone_selected_ratio` | 0.719338 | 4.171249 | 0.770751 | [scores](results/zone_weights_20261017/summary.csv) |
| 15 | `zone_weights_20261017/zone_fraction_0.5_ratio` | 0.719453 | 4.171662 | 0.770984 | [scores](results/zone_weights_20261017/summary.csv) |
| 16 | `zone_weights_20261017/zone_fraction_0.75_uniform` | 0.719504 | 4.169568 | 0.773465 | [scores](results/zone_weights_20261017/summary.csv) |
| 17 | `zone_weights_20261017/zone_fraction_0.25_ratio` | 0.719519 | 4.172039 | 0.770844 | [scores](results/zone_weights_20261017/summary.csv) |
| 18 | `zone_weights_20261017/common_selected_ratio` | 0.719632 | 4.172963 | 0.770844 | [scores](results/zone_weights_20261017/summary.csv) |
| 19 | `zone_weights_20261017/zone_fraction_0.75_ratio` | 0.719828 | 4.173853 | 0.771328 | [scores](results/zone_weights_20261017/summary.csv) |
| 20 | `zone_weights_20261017/pretest_selector` | 0.719952 | 4.173446 | 0.770751 | [scores](results/zone_weights_20261017/summary.csv) |
| 21 | `zone_weights_20261017/global_ratio` | 0.720002 | 4.174800 | 0.770874 | [scores](results/zone_weights_20261017/summary.csv) |
| 22 | `monthly_ratio_20261017/blend_ratio` | 0.720436 | 4.177390 | 0.782466 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 23 | `zone_weights_20261017/zone_uniform` | 0.720485 | 4.175361 | 0.773259 | [scores](results/zone_weights_20261017/summary.csv) |
| 24 | `monthly_quantiles_20261017/blend_uniform` | 0.720494 | 4.176488 | 0.777091 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 25 | `zone_weights_20261017/zone_ratio` | 0.720644 | 4.178603 | 0.771860 | [scores](results/zone_weights_20261017/summary.csv) |
| 26 | `monthly_quantiles_20261017/validation_selector` | 0.721285 | 4.181172 | 0.777091 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 27 | `monthly_quantiles_20261017/blend_ratio` | 0.721680 | 4.182921 | 0.771350 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 28 | `monthly_ratio_20261017/validation_selector` | 0.722556 | 4.188699 | 0.782466 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 29 | `minimize_expanding_20261017/fixed_cnn2d_analog_uniform` | 0.722965 | 4.190350 | 0.769779 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 30 | `minimize_expanding_20261017/fixed_cnn2d_analog_ratio` | 0.724033 | 4.198922 | 0.767532 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 31 | `minimize_expanding_20261017/fixed_linear_compact_ratio` | 0.724553 | 4.201780 | 0.766425 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 32 | `monthly_ratio_20261017/blend_uniform` | 0.724592 | 4.198074 | 0.790942 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 33 | `minimize_expanding_20261017/fixed_linear_compact_uniform` | 0.725535 | 4.206535 | 0.768657 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 34 | `ensemble_normalization_20261017/linear_compact__fixed` | 0.725707 | 4.207480 | 0.768971 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 35 | `minimize_expanding_20261017/fixed_linear_rich_ratio` | 0.726839 | 4.211747 | 0.775835 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 36 | `minimize_expanding_20261017/fixed_mlp_compact_uniform` | 0.726885 | 4.218104 | 0.764570 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 37 | `ensemble_normalization_20261017/cnn2d_analog__fixed` | 0.727906 | 4.224506 | 0.770528 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 38 | `minimize_expanding_20261017/fixed_cnn_rich_uniform` | 0.727913 | 4.220295 | 0.784178 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 39 | `ensemble_normalization_20261017/linear_rich__fixed` | 0.728020 | 4.217723 | 0.777765 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 40 | `minimize_expanding_20261017/fixed_additive_compact_ratio` | 0.728128 | 4.222913 | 0.781403 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 41 | `minimize_expanding_20261017/fixed_linear_rich_uniform` | 0.728609 | 4.220166 | 0.781371 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 42 | `minimize_expanding_20261017/fixed_additive_compact_uniform` | 0.729586 | 4.230943 | 0.786101 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 43 | `minimize_expanding_20261017/fixed_mlp_compact_ratio` | 0.730118 | 4.239394 | 0.782847 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 44 | `ensemble_normalization_20261017/additive_compact__fixed` | 0.730142 | 4.233447 | 0.786101 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 45 | `monthly_ratio_20261017/linear_compact_ratio` | 0.730329 | 4.235838 | 0.772705 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 46 | `monthly_quantiles_20261017/linear_compact_ratio` | 0.730386 | 4.234448 | 0.775540 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 47 | `monthly_quantiles_20261017/linear_compact_uniform` | 0.730864 | 4.237331 | 0.774076 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 48 | `minimize_expanding_20261017/fixed_cnn_rich_ratio` | 0.732949 | 4.249768 | 0.814560 | [scores](results/minimize_expanding_20261017/summary.csv) |
| 49 | `monthly_ratio_20261017/cnn2d_analog_ratio` | 0.733849 | 4.253503 | 0.785283 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 50 | `monthly_ratio_20261017/linear_rich_ratio` | 0.734376 | 4.253784 | 0.789313 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 51 | `monthly_ratio_20261017/linear_compact_uniform` | 0.734480 | 4.258332 | 0.776661 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 52 | `monthly_quantiles_20261017/linear_rich_ratio` | 0.734666 | 4.253466 | 0.792048 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 53 | `ensemble_normalization_20261017/mlp_compact__fixed` | 0.734969 | 4.266283 | 0.783522 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 54 | `monthly_quantiles_20261017/cnn2d_analog_ratio` | 0.735008 | 4.258815 | 0.788457 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 55 | `monthly_quantiles_20261017/linear_rich_uniform` | 0.735547 | 4.260377 | 0.788426 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 56 | `monthly_ratio_20261017/additive_compact_ratio` | 0.736176 | 4.271318 | 0.795335 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 57 | `monthly_quantiles_20261017/cnn2d_analog_uniform` | 0.736778 | 4.269359 | 0.787635 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 58 | `monthly_ratio_20261017/additive_compact_uniform` | 0.736778 | 4.274898 | 0.784684 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 59 | `monthly_quantiles_20261017/additive_compact_uniform` | 0.737004 | 4.275943 | 0.803358 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 60 | `monthly_quantiles_20261017/additive_compact_ratio` | 0.737317 | 4.277164 | 0.797113 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 61 | `monthly_ratio_20261017/cnn2d_analog_uniform` | 0.738069 | 4.277404 | 0.791745 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 62 | `normalized_validation_20261017/linear` | 0.738303 | 4.273508 | 0.798840 | [scores](results/normalized_validation_20261017/summary.csv) |
| 63 | `ensemble_normalization_20261017/cnn_rich__fixed` | 0.738978 | 4.286494 | 0.792937 | [scores](results/ensemble_normalization_20261017/summary.csv) |
| 64 | `monthly_ratio_20261017/linear_rich_uniform` | 0.739210 | 4.279253 | 0.798840 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 65 | `monthly_ratio_20261017/analog_shrink_fixed` | 0.739302 | 4.286808 | 0.788008 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 66 | `normalized_validation_20261017/linear_quantile_median` | 0.739333 | 4.278685 | 0.802222 | [scores](results/normalized_validation_20261017/summary.csv) |
| 67 | `monthly_ratio_20261017/ridge_compact_uniform` | 0.746022 | 4.326771 | 0.783483 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 68 | `monthly_quantiles_20261017/mlp_compact_uniform` | 0.746533 | 4.333533 | 0.790561 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 69 | `monthly_ratio_20261017/ridge_compact_ratio` | 0.746760 | 4.331400 | 0.797110 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 70 | `monthly_ratio_20261017/mlp_compact_ratio` | 0.748934 | 4.346632 | 0.833301 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 71 | `monthly_quantiles_20261017/mlp_compact_ratio` | 0.750620 | 4.361057 | 0.785073 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 72 | `monthly_ratio_20261017/ridge_rich_ratio` | 0.752705 | 4.362447 | 0.809196 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 73 | `normalized_validation_20261017/additive` | 0.753150 | 4.366023 | 0.804090 | [scores](results/normalized_validation_20261017/summary.csv) |
| 74 | `monthly_ratio_20261017/ridge_rich_uniform` | 0.754628 | 4.373205 | 0.813799 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 74 | `normalized_validation_20261017/ridge` | 0.754628 | 4.373205 | 0.813799 | [scores](results/normalized_validation_20261017/summary.csv) |
| 76 | `monthly_ratio_20261017/mlp_compact_uniform` | 0.757276 | 4.397425 | 0.831850 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 77 | `normalized_validation_20261017/hierarchical` | 0.762695 | 4.419719 | 0.806269 | [scores](results/normalized_validation_20261017/summary.csv) |
| 78 | `monthly_ratio_20261017/cnn_rich_ratio` | 0.762879 | 4.424082 | 0.865127 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 79 | `normalization_20261017/s5_fixed` | 0.762926 | 4.422713 | 0.803421 | [scores](results/normalization_20261017/summary.csv) |
| 80 | `normalization_20261017/s4d_fixed` | 0.763497 | 4.430358 | 0.802067 | [scores](results/normalization_20261017/summary.csv) |
| 81 | `normalized_validation_20261017/direct` | 0.765104 | 4.433215 | 0.807193 | [scores](results/normalized_validation_20261017/summary.csv) |
| 82 | `monthly_quantiles_20261017/cnn_rich_ratio` | 0.765939 | 4.440318 | 0.849123 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 83 | `monthly_ratio_20261017/cnn_rich_uniform` | 0.767810 | 4.452897 | 0.843023 | [scores](results/monthly_ratio_20261017/summary.csv) |
| 84 | `monthly_quantiles_20261017/cnn_rich_uniform` | 0.768762 | 4.454864 | 0.873898 | [scores](results/monthly_quantiles_20261017/summary.csv) |
| 85 | `normalization_20261017/compact_fixed` | 0.769238 | 4.458661 | 0.821142 | [scores](results/normalization_20261017/summary.csv) |
| 86 | `normalized_validation_20261017/cnn` | 0.773003 | 4.488442 | 0.846016 | [scores](results/normalized_validation_20261017/summary.csv) |
| 87 | `normalization_20261017/validation_selector` | 0.774028 | 4.490647 | 0.821142 | [scores](results/normalization_20261017/summary.csv) |
| 88 | `normalized_validation_20261017/mlp` | 0.776817 | 4.476401 | 0.900353 | [scores](results/normalized_validation_20261017/summary.csv) |
| 89 | `ssm_monthly_20261017/tcn` | 0.778942 | 4.512701 | 0.859556 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 90 | `normalization_20261017/s4d_standard` | 0.782518 | 4.530352 | 0.845712 | [scores](results/normalization_20261017/summary.csv) |
| 91 | `ssm_monthly_20261017/s4d` | 0.785554 | 4.551869 | 0.845712 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 92 | `ssm_monthly_20261017/gru` | 0.785659 | 4.551947 | 0.844647 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 93 | `ssm_monthly_20261017/validation_selector` | 0.787638 | 4.566457 | 0.857846 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 94 | `ssm_monthly_20261017/s5` | 0.788695 | 4.568278 | 0.857846 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 95 | `normalization_20261017/s5_standard` | 0.789864 | 4.576548 | 0.857846 | [scores](results/normalization_20261017/summary.csv) |
| 96 | `riccardo_comparison_20261017/riccardo_reduced` | 0.796834 | 4.615017 | 0.866863 | [scores](results/riccardo_comparison_20261017/summary.csv) |
| 97 | `normalization_20261017/s4d_day_input` | 0.796951 | 4.625565 | 0.855971 | [scores](results/normalization_20261017/summary.csv) |
| 98 | `normalization_20261017/compact_standard` | 0.797334 | 4.628212 | 0.872864 | [scores](results/normalization_20261017/summary.csv) |
| 99 | `ssm_monthly_20261017/compact` | 0.797603 | 4.629994 | 0.872864 | [scores](results/ssm_monthly_20261017/summary.csv) |
| 100 | `normalization_20261017/s5_day_input` | 0.799327 | 4.640775 | 0.843552 | [scores](results/normalization_20261017/summary.csv) |
| 101 | `normalization_20261017/s4d_day_reversible` | 0.802287 | 4.662785 | 0.835269 | [scores](results/normalization_20261017/summary.csv) |
| 102 | `riccardo_comparison_20261017/riccardo_raw_hourly` | 0.803239 | 4.658695 | 0.918460 | [scores](results/riccardo_comparison_20261017/summary.csv) |
| 103 | `normalization_20261017/s5_day_reversible` | 0.805933 | 4.681590 | 0.886023 | [scores](results/normalization_20261017/summary.csv) |
| 104 | `normalization_20261017/compact_day_input` | 0.807165 | 4.687277 | 0.908065 | [scores](results/normalization_20261017/summary.csv) |
| 105 | `linear_s4d_20261017/learned_frequencies` | 0.810498 | 4.698031 | 0.882753 | [scores](results/linear_s4d_20261017/summary.csv) |
| 105 | `linear_s4d_20261017/validation_selector` | 0.810498 | 4.698031 | 0.882753 | [scores](results/linear_s4d_20261017/summary.csv) |
| 107 | `normalization_20261017/compact_day_reversible` | 0.812202 | 4.714042 | 0.894793 | [scores](results/normalization_20261017/summary.csv) |
| 108 | `linear_s4d_20261017/fixed_frequencies` | 0.817796 | 4.741722 | 0.887342 | [scores](results/linear_s4d_20261017/summary.csv) |
| 109 | `normalized_validation_20261017/interpolation` | 1.000000 | 5.814817 | 1.000000 | [scores](results/normalized_validation_20261017/summary.csv) |
| 110 | `deterministic/hourly_repeat` | 1.412642 | 8.253379 | 1.508446 | [scores](results/monthly_baseline_mae.csv) |

## Exploratory pretrained comparison: September only

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `flowstate_20261017/conditional_control` | 5.454209 | 0.769247 | Full August; pretrained overlap unaudited |
| 2 | `flowstate_20261017/conditional_flowstate` | 5.461774 | 0.770314 | Full August; pretrained overlap unaudited |
| 3 | `flowstate_20261017/interpolation` | 7.090321 | 1.000000 | Full August; pretrained overlap unaudited |
| 4 | `flowstate_20261017/flowstate_projected_only` | 7.109678 | 1.002730 | Full August; pretrained overlap unaudited |
| 5 | `flowstate_20261017/hourly_mean` | 10.695366 | 1.508446 | Full August; pretrained overlap unaudited |

## Fixed-September history diagnostic: average over training histories

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `expanding_vast/history_average_ridge` | 5.292538 | 0.746445 | Diagnostic average, not one fitted model or independent folds |
| 2 | `expanding_vast/history_average_direct` | 5.419468 | 0.764347 | Diagnostic average, not one fitted model or independent folds |
| 3 | `expanding_vast/history_average_hierarchical` | 5.439638 | 0.767192 | Diagnostic average, not one fitted model or independent folds |
| 4 | `expanding_vast/history_average_interpolation` | 7.090321 | 1.000000 | Diagnostic average, not one fitted model or independent folds |

## Fixed-September training-history diagnostic

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `expanding_simple_5090/fixed_history_9_linear` | 5.167793 | 0.728852 | Eight histories reuse September; not eight independent tests |
| 2 | `expanding_simple_5090/fixed_history_7_linear` | 5.184737 | 0.731241 | Eight histories reuse September; not eight independent tests |
| 3 | `linear_quantiles_vast/fixed_history_9_linear_quantile` | 5.185206 | 0.731308 | Joint MAE/pinball selection; repeated September target set |
| 4 | `expanding_simple_5090/fixed_history_6_cnn` | 5.186334 | 0.731467 | Eight histories reuse September; not eight independent tests |
| 5 | `linear_quantiles_vast/fixed_history_8_linear_quantile` | 5.188030 | 0.731706 | Joint MAE/pinball selection; repeated September target set |
| 6 | `linear_quantiles_vast/fixed_history_7_linear_quantile` | 5.199417 | 0.733312 | Joint MAE/pinball selection; repeated September target set |
| 7 | `expanding_simple_5090/fixed_history_10_linear` | 5.200313 | 0.733438 | Eight histories reuse September; not eight independent tests |
| 8 | `linear_quantiles_vast/fixed_history_6_linear_quantile` | 5.200705 | 0.733494 | Joint MAE/pinball selection; repeated September target set |
| 9 | `expanding_simple_5090/fixed_history_5_linear` | 5.204933 | 0.734090 | Eight histories reuse September; not eight independent tests |
| 10 | `expanding_simple_5090/fixed_history_6_linear` | 5.207939 | 0.734514 | Eight histories reuse September; not eight independent tests |
| 11 | `expanding_simple_5090/fixed_history_8_linear` | 5.214049 | 0.735376 | Eight histories reuse September; not eight independent tests |
| 12 | `linear_quantiles_vast/fixed_history_10_linear_quantile` | 5.226105 | 0.737076 | Joint MAE/pinball selection; repeated September target set |
| 13 | `expanding_simple_5090/fixed_history_4_cnn` | 5.228618 | 0.737430 | Eight histories reuse September; not eight independent tests |
| 14 | `expanding_simple_5090/fixed_history_8_cnn` | 5.235246 | 0.738365 | Eight histories reuse September; not eight independent tests |
| 15 | `expanding_simple_5090/fixed_history_5_cnn` | 5.238621 | 0.738841 | Eight histories reuse September; not eight independent tests |
| 16 | `expanding_simple_5090/fixed_history_3_cnn` | 5.241403 | 0.739233 | Eight histories reuse September; not eight independent tests |
| 17 | `expanding_simple_5090/fixed_history_5_mlp` | 5.242400 | 0.739374 | Eight histories reuse September; not eight independent tests |
| 18 | `expanding_simple_5090/fixed_history_7_mlp` | 5.243471 | 0.739525 | Eight histories reuse September; not eight independent tests |
| 19 | `expanding_simple_5090/fixed_history_6_mlp` | 5.245217 | 0.739771 | Eight histories reuse September; not eight independent tests |
| 20 | `expanding_simple_5090/fixed_history_7_ridge` | 5.270116 | 0.743283 | Eight histories reuse September; not eight independent tests |
| 21 | `expanding_simple_5090/fixed_history_8_ridge` | 5.272215 | 0.743579 | Eight histories reuse September; not eight independent tests |
| 22 | `expanding_simple_5090/fixed_history_9_ridge` | 5.273739 | 0.743794 | Eight histories reuse September; not eight independent tests |
| 23 | `expanding_simple_5090/fixed_history_5_additive` | 5.279203 | 0.744565 | Eight histories reuse September; not eight independent tests |
| 24 | `linear_quantiles_vast/fixed_history_4_linear_quantile` | 5.279862 | 0.744658 | Joint MAE/pinball selection; repeated September target set |
| 25 | `expanding_simple_5090/fixed_history_3_linear` | 5.279869 | 0.744659 | Eight histories reuse September; not eight independent tests |
| 26 | `expanding_simple_5090/fixed_history_5_ridge` | 5.281218 | 0.744849 | Eight histories reuse September; not eight independent tests |
| 27 | `expanding_simple_5090/fixed_history_6_ridge` | 5.281303 | 0.744861 | Eight histories reuse September; not eight independent tests |
| 28 | `expanding_simple_5090/fixed_history_4_additive` | 5.286151 | 0.745545 | Eight histories reuse September; not eight independent tests |
| 29 | `expanding_simple_5090/fixed_history_10_ridge` | 5.294222 | 0.746683 | Eight histories reuse September; not eight independent tests |
| 30 | `expanding_simple_5090/fixed_history_6_additive` | 5.298453 | 0.747280 | Eight histories reuse September; not eight independent tests |
| 31 | `expanding_simple_5090/fixed_history_4_linear` | 5.299555 | 0.747435 | Eight histories reuse September; not eight independent tests |
| 32 | `expanding_simple_5090/fixed_history_8_additive` | 5.302933 | 0.747912 | Eight histories reuse September; not eight independent tests |
| 33 | `expanding_simple_5090/fixed_history_9_additive` | 5.309010 | 0.748769 | Eight histories reuse September; not eight independent tests |
| 34 | `expanding_simple_5090/fixed_history_3_additive` | 5.309096 | 0.748781 | Eight histories reuse September; not eight independent tests |
| 35 | `linear_quantiles_vast/fixed_history_5_linear_quantile` | 5.322369 | 0.750653 | Joint MAE/pinball selection; repeated September target set |
| 36 | `expanding_simple_5090/fixed_history_4_ridge` | 5.322553 | 0.750679 | Eight histories reuse September; not eight independent tests |
| 37 | `expanding_simple_5090/fixed_history_8_mlp` | 5.324037 | 0.750888 | Eight histories reuse September; not eight independent tests |
| 38 | `expanding_simple_5090/fixed_history_9_cnn` | 5.331246 | 0.751905 | Eight histories reuse September; not eight independent tests |
| 39 | `expanding_simple_5090/fixed_history_9_mlp` | 5.336133 | 0.752594 | Eight histories reuse September; not eight independent tests |
| 40 | `expanding_simple_5090/fixed_history_10_additive` | 5.344595 | 0.753787 | Eight histories reuse September; not eight independent tests |
| 41 | `expanding_simple_5090/fixed_history_3_ridge` | 5.344937 | 0.753836 | Eight histories reuse September; not eight independent tests |
| 42 | `expanding_simple_5090/fixed_history_10_mlp` | 5.363484 | 0.756451 | Eight histories reuse September; not eight independent tests |
| 43 | `expanding_simple_5090/fixed_history_3_mlp` | 5.373867 | 0.757916 | Eight histories reuse September; not eight independent tests |
| 44 | `expanding_simple_5090/fixed_history_10_cnn` | 5.377696 | 0.758456 | Eight histories reuse September; not eight independent tests |
| 45 | `linear_quantiles_vast/fixed_history_3_linear_quantile` | 5.377900 | 0.758485 | Joint MAE/pinball selection; repeated September target set |
| 46 | `expanding_simple_5090/fixed_history_7_additive` | 5.378633 | 0.758588 | Eight histories reuse September; not eight independent tests |
| 47 | `expanding_simple_5090/fixed_history_7_cnn` | 5.402854 | 0.762004 | Eight histories reuse September; not eight independent tests |
| 48 | `expanding_simple_5090/fixed_history_4_mlp` | 5.417071 | 0.764009 | Eight histories reuse September; not eight independent tests |
| 49 | `expanding_simple_5090/fixed_history_10_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 50 | `expanding_simple_5090/fixed_history_3_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 51 | `expanding_simple_5090/fixed_history_4_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 52 | `expanding_simple_5090/fixed_history_5_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 53 | `expanding_simple_5090/fixed_history_6_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 54 | `expanding_simple_5090/fixed_history_7_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 55 | `expanding_simple_5090/fixed_history_8_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |
| 56 | `expanding_simple_5090/fixed_history_9_interpolation` | 7.090321 | 1.000000 | Eight histories reuse September; not eight independent tests |

## Historical 60-day holdout: 2026-08-02..2026-09-30

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `catboost_search_gpu1/selected_ensemble` | 4.779861 | 0.715271 | 2026-06-03..2026-08-01 |
| 2 | `projection_search_gpu1/selected_ensemble` | 4.780831 | 0.715416 | 2026-06-03..2026-08-01 |
| 3 | `zone_search_gpu1/selected_ensemble` | 4.781314 | 0.715489 | 2026-06-03..2026-08-01 |
| 4 | `rich_search_gpu1/selected_ensemble` | 4.787278 | 0.716381 | 2026-06-03..2026-08-01 |
| 5 | `search_gpu1/selected_ensemble` | 5.036200 | 0.753630 | 2026-06-03..2026-08-01 |

## September 2026 only

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `monthly_revin_gpu1/selected_pipeline` | 5.091086 | 0.718033 | Full August |
| 2 | `monthly_joint_blend_gpu1/selected_pipeline` | 5.091209 | 0.718051 | Full August |
| 3 | `monthly_graph_gpu1/selected_pipeline` | 5.092270 | 0.718200 | Full August |
| 4 | `monthly_cat_gpu1/selected_pipeline` | 5.122793 | 0.722505 | Full August |
| 5 | `monthly_base_gpu1/selected_pipeline` | 5.125800 | 0.722929 | Full August |
| 6 | `parallel_2d_tuning_20261017/analog90` | 5.177024 | 0.730154 | Full August; differing budget |
| 7 | `simple_5090/linear` | 5.209794 | 0.734776 | August1–15 |
| 8 | `quantile_2d_cnn_20261017/analog90_1` | 5.216651 | 0.735743 | August1–15; cohort winners |
| 9 | `simple_5090/cnn` | 5.222494 | 0.736567 | August1–15 |
| 10 | `simple_5090/additive` | 5.255107 | 0.741166 | August1–15 |
| 11 | `parallel_2d_tuning_20261017/context` | 5.261885 | 0.742122 | Full August; differing budget |
| 12 | `hierarchical_hpo_lags_vast/parent_selected` | 5.282777 | 0.745069 | August1–15 |
| 13 | `hierarchical_hpo_vast/candidate_1` | 5.282777 | 0.745069 | August1–15 |
| 14 | `quantile_2d_cnn_20261017/analog90_0` | 5.336198 | 0.752603 | August1–15; cohort winners |
| 15 | `simple_5090/mlp` | 5.343947 | 0.753696 | August1–15 |
| 16 | `parallel_2d_tuning_20261017/analog7` | 5.346014 | 0.753988 | Full August; differing budget |
| 17 | `hierarchical_vast/candidate_7` | 5.352400 | 0.754888 | August1–15 |
| 18 | `quantile_2d_cnn_20261017/raw_1` | 5.427657 | 0.765502 | August1–15; cohort winners |
| 19 | `parallel_2d_zone_cnn_20261017/history_1` | 5.431127 | 0.765992 | Full August |
| 20 | `parallel_2d_tuning_20261017/control` | 5.458611 | 0.769868 | Full August; differing budget |
| 21 | `quantile_2d_cnn_20261017/raw_0` | 5.462844 | 0.770465 | August1–15; cohort winners |
| 22 | `joint_zone_cnn_20261017/history_1` | 5.485715 | 0.773691 | Full August |
| 23 | `parallel_2d_tuning_20261017/clock` | 5.487315 | 0.773916 | Full August; differing budget |
| 24 | `parallel_2d_tuning_20261017/residual` | 5.536894 | 0.780909 | Full August; differing budget |
| 25 | `parallel_2d_tuning_20261017/width16` | 5.538923 | 0.781195 | Full August; differing budget |
| 26 | `simple_joint_zone_cnn_20261017/history_1` | 5.539996 | 0.781346 | Full August |
| 27 | `interpretable_vast/additive_flat_history1_alpha10` | 5.688563 | 0.802300 | August1–15 |
| 28 | `four_frequency_20261017/pca4` | 5.827434 | 0.821886 | Full August |
| 29 | `joint_zone_cnn_20261017/history_0` | 5.862788 | 0.826872 | Full August |
| 30 | `four_frequency_20261017/pca16` | 5.887545 | 0.830364 | Full August |
| 31 | `four_frequency_20261017/pca8` | 5.887884 | 0.830412 | Full August |
| 32 | `parallel_2d_zone_cnn_20261017/history_0` | 5.903767 | 0.832652 | Full August |
| 33 | `simple_joint_zone_cnn_20261017/history_0` | 5.921824 | 0.835198 | Full August |
| 34 | `segmented_frequency_20261017/global8` | 5.947649 | 0.838841 | Full August |
| 35 | `four_frequency_20261017/fourier4_freephase` | 5.977431 | 0.843041 | Full August |
| 36 | `segmented_frequency_20261017/seasonal` | 6.124439 | 0.863775 | Full August |
| 37 | `segmented_frequency_20261017/constant` | 6.134637 | 0.865213 | Full August |
| 38 | `four_frequency_20261017/fourier4` | 6.210906 | 0.875970 | Full August |
| 39 | `segmented_frequency_20261017/noon` | 6.397598 | 0.902300 | Full August |
| 40 | `four_frequency_20261017/interpolation` | 7.090321 | 1.000000 | Full August |
| 41 | `segmented_frequency_20261017/interpolation` | 7.090321 | 1.000000 | Full August |

## Superseded SSM pilot: September only

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `ssm_pilot_20261017/tcn` | 5.452077 | 0.768946 | Full August; later eight-fold campaign supersedes pilot |
| 2 | `ssm_pilot_20261017/s4d` | 5.497365 | 0.775334 | Full August; later eight-fold campaign supersedes pilot |
| 3 | `ssm_pilot_20261017/gru` | 5.517535 | 0.778178 | Full August; later eight-fold campaign supersedes pilot |
| 4 | `ssm_pilot_20261017/compact` | 5.591747 | 0.788645 | Full August; later eight-fold campaign supersedes pilot |
| 5 | `ssm_pilot_20261017/s5` | 5.603856 | 0.790353 | Full August; later eight-fold campaign supersedes pilot |

## Superseded SSM pilot: four earlier rolling folds

Ranks apply only within this scope. These rows do not enter the primary eight-fold promotion.

| Rank within scope | Recipe | MAE | Ratio to same-scope interpolation | Selection |
|---:|---|---:|---:|---|
| 1 | `ssm_pilot_20261017/rolling_gru` | 4.356461 | Unavailable | Fixed width32/lr.001/seed17; different folds, no eight-month ratio |
| 2 | `ssm_pilot_20261017/rolling_s4d` | 4.357039 | Unavailable | Fixed width32/lr.001/seed17; different folds, no eight-month ratio |
| 3 | `ssm_pilot_20261017/rolling_compact` | 4.359954 | Unavailable | Fixed width32/lr.001/seed17; different folds, no eight-month ratio |
| 4 | `ssm_pilot_20261017/rolling_tcn` | 4.386337 | Unavailable | Fixed width32/lr.001/seed17; different folds, no eight-month ratio |
| 5 | `ssm_pilot_20261017/rolling_s5` | 4.391647 | Unavailable | Fixed width32/lr.001/seed17; different folds, no eight-month ratio |

## Unranked diagnostics and coverage

10 target-derived frequency/PCA oracle results are listed in oracle_diagnostics.csv. They measure representation bounds and cannot become champions. Exploratory pretrained FlowState results have a separate scope and are excluded from eligible challenge promotion. Fixed-September history sizes reuse one test month. Smokes, superseded pilots, incomplete results and all-data deployment refits have no held-out primary rank.

Inventory and full data: [primary_ranking.csv](results/model_ranking_20261017/primary_ranking.csv), [all reported rows and aliases](results/model_ranking_20261017/all_reported_models.csv), [monthly scores](results/model_ranking_20261017/monthly_scores.csv), [historical scopes](results/model_ranking_20261017/historical_ranking.csv), [candidate catalogue](results/model_ranking_20261017/candidate_catalog.csv), [campaign inventory](results/model_ranking_20261017/campaign_inventory.csv), [provenance/audit](results/model_ranking_20261017/audit.json).

Reproduce with `python3 rank_models.py`. This consolidates saved metrics and recomputes their aggregate scores; it performs no new fitting, model prediction or recipe tuning. All results remain developmental. Original evaluation and deployment artifacts retain their provenance.
