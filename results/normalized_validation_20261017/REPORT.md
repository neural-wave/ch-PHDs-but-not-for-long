# Monthly validation with interpolation normalization

Verified 17 October 2026. Quick rescore of completed developmental backtests; no retraining.

Protocol: [VALIDATION_PROTOCOL.md](../../VALIDATION_PROTOCOL.md). Primary score is the arithmetic mean of the eight monthly MAE / mean-corrected-linear-interpolation-MAE ratios. Raw MAE is pooled by target count. Lower is better. Ratio 1 is interpolation parity.

| Method | Pooled MAE (EUR/MWh) | Mean monthly ratio | Reduction vs interpolation | Worst monthly ratio |
|---|---:|---:|---:|---:|
| linear | 4.273508 | 0.738303 | 26.17% | 0.798840 |
| linear_quantile_median | 4.278685 | 0.739333 | 26.07% | 0.802222 |
| additive | 4.366023 | 0.753150 | 24.69% | 0.804090 |
| ridge | 4.373205 | 0.754628 | 24.54% | 0.813799 |
| hierarchical | 4.419719 | 0.762695 | 23.73% | 0.806269 |
| direct | 4.433215 | 0.765104 | 23.49% | 0.807193 |
| cnn | 4.488442 | 0.773003 | 22.70% | 0.846016 |
| mlp | 4.476401 | 0.776817 | 22.32% | 0.900353 |
| interpolation | 5.814817 | 1.000000 | 0.00% | 1.000000 |

## Month-by-month normalized errors

| Test month | Interpolation MAE | additive | cnn | direct | hierarchical | interpolation | linear | linear_quantile_median | mlp | ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-02 | 3.8489 | 0.7911 | 0.8140 | 0.8072 | 0.8030 | 1.0000 | 0.7988 | 0.8022 | 0.9004 | 0.7933 |
| 2026-03 | 6.5205 | 0.7443 | 0.7423 | 0.7679 | 0.7675 | 1.0000 | 0.7296 | 0.7262 | 0.7740 | 0.7411 |
| 2026-04 | 5.8315 | 0.8041 | 0.8460 | 0.8037 | 0.8063 | 1.0000 | 0.7933 | 0.7994 | 0.8406 | 0.8138 |
| 2026-05 | 5.6456 | 0.7644 | 0.8062 | 0.7779 | 0.7784 | 1.0000 | 0.7328 | 0.7298 | 0.7736 | 0.7741 |
| 2026-06 | 5.8390 | 0.6966 | 0.6995 | 0.7815 | 0.7748 | 1.0000 | 0.6906 | 0.6936 | 0.6999 | 0.7245 |
| 2026-07 | 5.3796 | 0.7478 | 0.7220 | 0.7143 | 0.7063 | 1.0000 | 0.7150 | 0.7153 | 0.7431 | 0.7238 |
| 2026-08 | 6.2380 | 0.7232 | 0.7143 | 0.7188 | 0.7240 | 1.0000 | 0.7128 | 0.7111 | 0.7264 | 0.7198 |
| 2026-09 | 7.0903 | 0.7538 | 0.8396 | 0.7496 | 0.7413 | 1.0000 | 0.7334 | 0.7371 | 0.7565 | 0.7467 |

## September-only studies

These methods have different model-selection campaigns and no eight-month estimate here. Do not rank their single-month ratios against the multi-month averages above.

| Method | September MAE (EUR/MWh) | September ratio | Selection |
|---|---:|---:|---|
| monthly_joint_blend | 5.091209 | 0.718051 | Full August |
| analog90_CNN | 5.177024 | 0.730154 | August 1-15 |
| quantile_campaign_analog90_1 | 5.216651 | 0.735743 | August 1-15; overall/cohort choice |
| quantile_campaign_analog90_0 | 5.336198 | 0.752603 | August 1-15; overall/cohort choice |

## Scope and verification

All nine methods cover the same eight test months and 241 days (92,544 quarter targets across four zones). Linear, additive, MLP, CNN, ridge and interpolation come from the completed 3,680-candidate expanding/fixed-history campaign on root@93.91.156.85:55340, /workspace/HACKATON/results/expanding_simple_5090. Only expanding folds enter this comparison. Direct/hierarchical neural models and linear quantiles come from their earlier eight-fold campaigns. Quantile selection uses its joint median-MAE/pinball objective; candidate budgets differ across campaigns, so this is a comparison of saved pipelines, not a controlled architecture ablation.

Independently recomputed both interpolation baselines from all official quarters. Checked saved monthly baselines, source hash, fold dates/counts, score consistency, simple-family validation argmins, daily MAE averages, per-zone averages and completed independent-audit receipts. Existing audits contain prediction/conservation/replay evidence. This rescore does not rerun all model predictions.

The denominator uses hour-centred linear interpolation with flat day-edge extension, followed by a constant correction within each hour to recover the supplied hourly mean. Uncorrected linear interpolation is also included as an explicitly named alternative denominator in monthly_metrics.csv. In September its MAE is 7.540344 versus 7.090321 after mean correction. Do not mix these denominators.

The reported average is mean(monthly model MAE / same-month interpolation MAE). It differs from pooled model MAE / pooled interpolation MAE, which remains a secondary column in summary.csv. Monthly ratios normalize baseline difficulty; they do not remove all seasonal distribution shifts or provide independent fold replications.

All months were available during previous development. Scores remain developmental. The newer analog and quantile CNNs have September-only evidence here. No full nested re-selection, new candidate fitting or deployment replacement was performed.

Reproduce: `python3 evaluate_normalized_validation.py`. Inputs and SHA256 receipts are listed in audit.json.
