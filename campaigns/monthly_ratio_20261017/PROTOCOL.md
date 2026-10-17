# Minimize average monthly interpolation-relative MAE

Authorized 17 October 2026 after adopting `VALIDATION_PROTOCOL.md`. Run on the existing Vast RTX5090 at root@93.91.156.85:55340, `/workspace/HACKATON`, `/venv/main/bin/python`. No local training, new paid instance, pretrained models, external data or deployment replacement.

The campaign imports its exact local-project dependency snapshot from `campaigns/monthly_ratio_20261017/sources/`. Preflight found the server's older compact-feature API lacked interpolation-history channels; isolation preserves old remote sources and fixes campaign reproducibility. All snapshot modules and campaign files are hashed in the manifest.

## Objective and comparison

The target is the equal-month arithmetic mean of `model_MAE / mean_corrected_linear_interpolation_MAE`. Use the established eight expanding folds, February–September 2026 tests, with the preceding month's first15days for stopping/configuration/blend selection and remaining days reserved. All dates/zones stay together. Train every fold from scratch; normalization/knots/weights use fitting dates only. Today's entire96-value four-zone hourly curve is known and conditions learned models. Observed quarters are strictly historical. Preserve each supplied hourly mean within0.0001EUR/MWh. September and other inspected months remain developmental.

Within one validation block its common interpolation denominator cannot change the MAE ranking. Therefore directly test a new **training objective**, in a paired comparison:

- Ordinary target-weighted MAE.
- Month-balanced relative MAE. For fitting month m, let B_m be its corrected interpolation MAE and n_m its fitting-day count. Give each fitting day weight proportional to `1 / (n_m * B_m)`, normalized to mean1. The resulting objective is proportional to mean(monthly training MAE/B_m). Compute B_m only from fitting targets. No test-month denominator enters training, stopping or selection.

Both objectives retain the same AdamW regularization, gradient clipping, schedules, seeds and model budget. Report raw training ratio separately from normalized-weight loss. This is empirical risk on the training distribution, not a guarantee of improved future monthly ratios.

## Fixed candidate budget

Six gradient-trained families: rich-feature linear; compact physical/history linear; compact additive hinges; compact MLP(width32,two hidden layers); rich temporal CNN(width16,two kernel5 convolutions); small2DCNN(width16,seven-day quarter/hour history,90-day analog residuals,kernel5 fusion) with an explicit96-hourly-value global projection. Compact inputs also include all96current hourly values, seven causal interpolation-error lags, analog correction and zone indicators. Rich inputs are the existing389-feature causal representation, already containing full-day hourly prices and analogs.

For every family/objective use learning rates0.001/0.003 and seeds17/43:48gradient fits per fold,384total. Max300epochs/patience35 for simple heads,450/patience50 for2DCNN. Training minibatch32. No choice may be added after inspecting outer results in this campaign.

Controls: rich/compact ridge at alpha1/10/100/1000/10000 under both ordinary and ratio weights, giving160closed-form fits; fixed causal analog residual shrinkage0/0.25/0.5/0.75/1 selected on the same block; corrected interpolation. A separate blend for each training objective chooses among individual winners, fixed global pairwise mixtures of the rich linear with other family winners (weights0.25/0.5/0.75), and an equal-weight top3validation ensemble. Blend selection uses only first15days; no zone-specific weights or fitted test adjustments. Also score the deployable rule that chooses the lowest preceding-block ratio among these recipes. Candidate budget is bounded; family-specific parameter counts and search spaces differ.

Save each family's per-fold selection before test scoring. Main table compares mean monthly ratio, pooled raw MAE, worst monthly ratio and all monthly/per-zone errors. Include the previous0.738303linear average as a historical same-fold reference, with its source provenance; it is not a newly paired run. Fixed-September history-size replicates are excluded.

## Verification and unattended completion

Short first/last-fold smoke covers all families/objectives, train-only weights/normalization, perturbation causality, missing-date history, finite gradients, exact means and serialized prediction replay. The full independent audit recomputes official source/baseline/fold/normalization/weight statistics, candidate validation scores, within-family argmins, blends/recipe selection, test scores/ratios and aggregate arithmetic from saved predictions. Replay each selected learned model. An asymmetric two-month fixture verifies that normalized weighting gives the mean of monthly ratios up to one common scale factor.

Execution-host Nightwatch wrapper reports heartbeats and actual success/nonzero exits/signals; completion requires the full independent audit and final report. Save source hashes, checkpoints, predictions, per-fold manifests, final report and a recovery archive under `results/monthly_ratio_20261017/`. No ETA-based success. Expected duration is initially20minutes and is revised only from measured work. No test-result-driven follow-up search is included in this launch.
