# Average monthly ratio optimization

Completed developmental eight-fold experiment. Each candidate was stopped/selected using only the preceding month's first15days. Ratio means MAE / same-month mean-corrected-linear-interpolation MAE. Lower is better. All point predictions preserve supplied hourly means.

| Recipe | Average monthly ratio | Pooled MAE (EUR/MWh) | Worst monthly ratio |
|---|---:|---:|---:|
| blend_ratio | 0.720436 | 4.177390 | 0.782466 |
| validation_selector | 0.722556 | 4.188699 | 0.782466 |
| blend_uniform | 0.724592 | 4.198074 | 0.790942 |
| linear_compact_ratio | 0.730329 | 4.235838 | 0.772705 |
| cnn2d_analog_ratio | 0.733849 | 4.253503 | 0.785283 |
| linear_rich_ratio | 0.734376 | 4.253784 | 0.789313 |
| linear_compact_uniform | 0.734480 | 4.258332 | 0.776661 |
| additive_compact_ratio | 0.736176 | 4.271318 | 0.795335 |
| additive_compact_uniform | 0.736778 | 4.274898 | 0.784684 |
| cnn2d_analog_uniform | 0.738069 | 4.277404 | 0.791745 |
| linear_rich_uniform | 0.739210 | 4.279253 | 0.798840 |
| analog_shrink_fixed | 0.739302 | 4.286808 | 0.788008 |
| ridge_compact_uniform | 0.746022 | 4.326771 | 0.783483 |
| ridge_compact_ratio | 0.746760 | 4.331400 | 0.797110 |
| mlp_compact_ratio | 0.748934 | 4.346632 | 0.833301 |
| ridge_rich_ratio | 0.752705 | 4.362447 | 0.809196 |
| ridge_rich_uniform | 0.754628 | 4.373205 | 0.813799 |
| mlp_compact_uniform | 0.757276 | 4.397425 | 0.831850 |
| cnn_rich_ratio | 0.762879 | 4.424082 | 0.865127 |
| cnn_rich_uniform | 0.767810 | 4.452897 | 0.843023 |
| interpolation_fixed | 1.000000 | 5.814817 | 1.000000 |

## Paired training-objective comparison

| Family | Ordinary training | Monthly-ratio training | Ratio minus ordinary |
|---|---:|---:|---:|
| linear_rich | 0.739210 | 0.734376 | -0.004833 |
| linear_compact | 0.734480 | 0.730329 | -0.004152 |
| additive_compact | 0.736778 | 0.736176 | -0.000601 |
| mlp_compact | 0.757276 | 0.748934 | -0.008342 |
| cnn_rich | 0.767810 | 0.762879 | -0.004931 |
| cnn2d_analog | 0.738069 | 0.733849 | -0.004220 |
| ridge_rich | 0.754628 | 0.752705 | -0.001923 |
| ridge_compact | 0.746022 | 0.746760 | +0.000738 |
| blend | 0.724592 | 0.720436 | -0.004155 |

## Monthly results

| Month | additive_compact_ratio | additive_compact_uniform | analog_shrink_fixed | blend_ratio | blend_uniform | cnn2d_analog_ratio | cnn2d_analog_uniform | cnn_rich_ratio | cnn_rich_uniform | interpolation_fixed | linear_compact_ratio | linear_compact_uniform | linear_rich_ratio | linear_rich_uniform | mlp_compact_ratio | mlp_compact_uniform | ridge_compact_ratio | ridge_compact_uniform | ridge_rich_ratio | ridge_rich_uniform | validation_selector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02 | 0.767524 | 0.767924 | 0.788008 | 0.760356 | 0.769263 | 0.785283 | 0.786567 | 0.792696 | 0.799963 | 1.000000 | 0.769693 | 0.775122 | 0.789313 | 0.798840 | 0.775334 | 0.780499 | 0.778298 | 0.780164 | 0.791384 | 0.793279 | 0.760356 |
| 2026-03 | 0.722700 | 0.715059 | 0.741687 | 0.708663 | 0.703734 | 0.718711 | 0.720270 | 0.737743 | 0.730207 | 1.000000 | 0.716960 | 0.715203 | 0.727947 | 0.725797 | 0.723332 | 0.746901 | 0.724993 | 0.726265 | 0.739384 | 0.741076 | 0.703734 |
| 2026-04 | 0.795335 | 0.784684 | 0.764317 | 0.782466 | 0.790942 | 0.770357 | 0.791745 | 0.865127 | 0.843023 | 1.000000 | 0.772705 | 0.776661 | 0.780339 | 0.793274 | 0.833301 | 0.831850 | 0.797110 | 0.783483 | 0.809196 | 0.813799 | 0.782466 |
| 2026-05 | 0.734111 | 0.739666 | 0.720351 | 0.720875 | 0.741810 | 0.731374 | 0.735103 | 0.766623 | 0.818485 | 1.000000 | 0.729114 | 0.750770 | 0.729095 | 0.732820 | 0.744457 | 0.792900 | 0.766482 | 0.767309 | 0.772758 | 0.774087 | 0.741810 |
| 2026-06 | 0.677266 | 0.680386 | 0.697066 | 0.662718 | 0.666725 | 0.678132 | 0.677202 | 0.733229 | 0.728533 | 1.000000 | 0.680986 | 0.681614 | 0.680952 | 0.696560 | 0.697471 | 0.686182 | 0.704451 | 0.706650 | 0.719738 | 0.724499 | 0.666725 |
| 2026-07 | 0.726394 | 0.728232 | 0.725700 | 0.700181 | 0.699082 | 0.715982 | 0.721244 | 0.719948 | 0.716605 | 1.000000 | 0.716006 | 0.717840 | 0.715710 | 0.715027 | 0.731499 | 0.723074 | 0.726237 | 0.726859 | 0.722875 | 0.723785 | 0.700181 |
| 2026-08 | 0.721352 | 0.733807 | 0.726024 | 0.699637 | 0.700994 | 0.722959 | 0.728326 | 0.724396 | 0.725590 | 1.000000 | 0.713654 | 0.715235 | 0.715913 | 0.712842 | 0.716771 | 0.719333 | 0.724461 | 0.725213 | 0.719422 | 0.719818 | 0.700994 |
| 2026-09 | 0.744727 | 0.744463 | 0.751259 | 0.728594 | 0.724183 | 0.747999 | 0.744095 | 0.763269 | 0.780077 | 1.000000 | 0.743510 | 0.743397 | 0.735742 | 0.738521 | 0.769303 | 0.777469 | 0.752047 | 0.752232 | 0.746883 | 0.746683 | 0.724183 |

The validation_selector row is a deployable rule chosen from prior validation results, not the best outer-test recipe. The lowest aggregate test result is exploratory evidence for future studies. These months have all been inspected during prior development, so results are not a blind benchmark. Search budgets/parameter counts vary across families. A lower aggregate alone does not establish statistical significance.

Ratio training weights are proportional to1/(training-month days × training-month interpolation MAE), normalized to mean1. No outer-test denominator enters fitting or selection. Calibration dates are reserved and unused for point-model fitting.

Historical same-fold linear reference: mean monthly ratio0.738303, pooledMAE4.273508 in results/normalized_validation_20261017/REPORT.md. It comes from the earlier broader search. Current paired uniform/ratio models share candidate spaces and are the appropriate control for the objective change.

Reproduction: campaigns/monthly_ratio_20261017/run.py; fixed budget and constraints in campaigns/monthly_ratio_20261017/PROTOCOL.md. Per-fold models, predictions, selections, source hashes, monthly_scores.csv, summary.csv and audit.json are retained. No deployment was replaced.
