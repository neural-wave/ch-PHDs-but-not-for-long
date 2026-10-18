# Riccardo branch under our monthly validation

Pinned branch `riccardo`, commit `1ed1423c5fa05586457347ab34e69b9bcfb23031`. Verified 17 October 2026.

Two fresh CNN pipelines use the same eight expanding folds, first-15-day epoch selection, full following-month tests, official targets and corrected-interpolation denominator as our saved nine pipelines. All learned weights and scales are fitted inside each outer training block. Days 16–month-end are reserved and unused; no new uncertainty intervals were requested.

| Method | Pooled MAE EUR/MWh | Mean monthly ratio | Worst month ratio |
|---|---:|---:|---:|
| linear | 4.273508 | 0.738303 | 0.798840 |
| linear_quantile_median | 4.278685 | 0.739333 | 0.802222 |
| additive | 4.366023 | 0.753150 | 0.804090 |
| ridge | 4.373205 | 0.754628 | 0.813799 |
| hierarchical | 4.419719 | 0.762695 | 0.806269 |
| direct | 4.433215 | 0.765104 | 0.807193 |
| cnn | 4.488442 | 0.773003 | 0.846016 |
| mlp | 4.476401 | 0.776817 | 0.900353 |
| riccardo_reduced | 4.615017 | 0.796834 | 0.866863 |
| riccardo_raw_hourly | 4.658695 | 0.803239 | 0.918460 |
| interpolation | 5.814817 | 1.000000 | 1.000000 |

## Monthly results

| Test month | Interpolation MAE | Reduced CNN MAE / ratio | Raw-hourly CNN MAE / ratio |
|---|---:|---:|---:|
| 2026-02 | 3.848893 | 3.221421 / 0.836973 | 3.193905 / 0.829824 |
| 2026-03 | 6.520536 | 4.990594 / 0.765366 | 5.259736 / 0.806642 |
| 2026-04 | 5.831511 | 5.055124 / 0.866863 | 5.356010 / 0.918460 |
| 2026-05 | 5.645602 | 4.756090 / 0.842442 | 4.773891 / 0.845595 |
| 2026-06 | 5.838993 | 4.377382 / 0.749681 | 4.205482 / 0.720241 |
| 2026-07 | 5.379630 | 4.159889 / 0.773267 | 4.112140 / 0.764391 |
| 2026-08 | 6.238029 | 4.730293 / 0.758299 | 4.686455 / 0.751272 |
| 2026-09 | 7.090321 | 5.543063 / 0.781779 | 5.597740 / 0.789490 |

## What was compared

The lowest committed aggregate August–September MAE is 5.087239411 for the reduced-profile CNN using refit month weights. The four-channel input contains day-centered hourly level, slope, curvature and boundary indicator, plus learned hour and weekday embeddings. Four separate zonal two-layer CNNs have widths 64/96/96/64. The README raw-hourly model scores 5.113040454, uses one unnormalized hourly input channel and no calendar embeddings. Both predict centered quarter residuals around the known hourly means, use Q1/Q4-weighted MAE (1.5,1,1,1.5), AdamW, gradient clipping 5, up to 150 epochs and patience 16. The imported per-zone configurations and original seeds are frozen.

Original scores are selection results over August and September together. Their LOMO difficulty scorers train on 11 of all 12 months; October–July month scores therefore depend on August/September targets despite restricting the subsequent minmax normalization to training months. This campaign recomputes LOMO scorers strictly within each outer training block and selects final epochs only on the next month’s first 15 days. Internal LOMO scorers retain the upstream held-month early stopping and can use later months inside the outer training block; they are a train-only weight estimator, not a chronological inner forecast benchmark. The clean adaptation is distinct from evaluating the original committed checkpoints.

Candidate/feature configurations were historically selected with inspected August/September data. The adaptation provides fresh chronological fits, not a blind estimate or a full nested rerun of their original architecture/feature search. Our saved pipelines have different search budgets and some use causal quarter-history/cross-zone inputs; compare complete pipelines, not architectures in isolation. All results remain developmental. No deployment model was replaced. External zone-identification and October files were excluded.

Earlier September-only blend/analog/quantile studies remain in `results/normalized_validation_20261017/september_only.csv`; they have no matched eight-month estimates and are not pooled into this table.

Audit: `independent_audit.json`; per-zone and daily results: `per_zone.csv`, `per_day_zone.csv`; selected states and predictions: `<test-month>/<zone>/*.pt,*.npz`. Source provenance: `campaigns/riccardo_comparison_20261017/upstream_receipt.json`. Execution host: `root@93.91.156.85:55340`, `/workspace/HACKATON`. Reproduce with `python compare_riccardo.py` then `python audit_riccardo_comparison.py` on that CUDA host.

## Direct comparison with our existing linear pipeline

| Imported variant | Rank among 11 pipelines | Change in primary ratio vs our linear | Change in pooled MAE | Months beating our linear |
|---|---:|---:|---:|---:|
| riccardo_reduced | 9 | +7.93% | +7.99% | 0/8 |
| riccardo_raw_hourly | 10 | +8.80% | +9.01% | 0/8 |

Positive changes mean higher error. Month wins are descriptive; dependent developmental folds do not support independent-fold inference.

Execution uses our existing CUDA/PyTorch 2.14.1+cu130 research environment. The upstream requirements pin PyTorch 2.8.0 and the committed run reports Apple MPS. Runtime/backend changes can affect optimization numerics; the smoke parity check compares original and adapted reduced-feature training in the same CUDA environment. We do not claim bitwise reproduction of the original MPS campaign.

## September on identical dates, with selection differences retained

This table includes earlier September-only studies at the user’s request. All rows score the same September targets, but the joint blend uses full August selection and the campaigns differ in search scope; this is not a uniform selection-budget comparison. Single-month winners do not replace the primary eight-month ranking.

| Method | September MAE EUR/MWh | September ratio | Evidence scope | Selection |
|---|---:|---:|---|---|
| monthly_joint_blend | 5.091209 | 0.718051 | September-only study; no eight-month estimate | Full August |
| analog90_CNN | 5.177024 | 0.730154 | September-only study; no eight-month estimate | August 1-15 |
| linear | 5.200313 | 0.733438 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| quantile_campaign_analog90_1 | 5.216651 | 0.735743 | September-only study; no eight-month estimate | August 1-15; overall/cohort choice |
| linear_quantile_median | 5.226105 | 0.737076 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| hierarchical | 5.256047 | 0.741299 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| ridge | 5.294222 | 0.746683 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| direct | 5.314717 | 0.749574 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| quantile_campaign_analog90_0 | 5.336198 | 0.752603 | September-only study; no eight-month estimate | August 1-15; overall/cohort choice |
| additive | 5.344595 | 0.753787 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| mlp | 5.363484 | 0.756451 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| riccardo_reduced | 5.543063 | 0.781779 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| riccardo_raw_hourly | 5.597740 | 0.789490 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| cnn | 5.953010 | 0.839597 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
| interpolation | 7.090321 | 1.000000 | September from eight-fold campaign | August 1–15; campaign budgets/configuration provenance differ |
