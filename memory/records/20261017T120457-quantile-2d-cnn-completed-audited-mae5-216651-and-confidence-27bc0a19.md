---
title: Quantile 2D CNN completed audited MAE5.216651 and confidence limitations 20261017
status: current
kind: result
recorded_at: '2026-10-17T12:04:57.097008+00:00'
sources:
- QUANTILE_2D_CNN_PROTOCOL.md
- quantile_cnn.py
- analyze_quantile_cnn.py
- analyze_cnn_confidence.py
- audit_quantile_cnn.py
- results/quantile_2d_cnn_20261017/metrics.json
- results/quantile_2d_cnn_20261017/audit.json
- results/quantile_2d_cnn_20261017/quantile_metrics.json
- results/quantile_2d_cnn_20261017/risk_models.json
- results/quantile_2d_cnn_20261017/confidence_metrics.json
- results/parallel_2d_tuning_20261017/audit.json
- results/parallel_2d_tuning_20261017/REPORT.md
experiment_id: quantile_2d_cnn_20261017
supersedes: []
---

# Completed quantile-CNN restart, 17 October 2026

## Authorization and result
User approved trying the discussed simple 2D CNN plus direct quantile training, with around5.2 MAE considered good enough. The24 extra values are the query day's supplied hourly prices. All computation ran on the existing authorized Vast RTX5090, root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python; no new rental, local research training, deployment refit or submission.

24 main fits and6 chronological held-out folds completed. Final artifact audit passed. August1–15 selects checkpoint/seed/config; August16–31 exclusively calibrates intervals and error-risk probes. September remains previously viewed developmental evaluation.

August-selected best in each predeclared input/loss cohort:
- raw, MAE only: trial0, SeptemberMAE5.462843825384081, August4.735578894615173.
- raw, MAE+pinball: trial9, September5.427656680603007, August4.7336554527282715.
- analog90, MAE only: trial13, September5.336198180744609, August4.509062469005585.
- analog90, MAE+pinball: trial23, September5.216650727743593, August4.519094824790955, LR.003, seed71, bestepoch78.
The quantile cohort winner achieves the user's approximate5.2 target. The overall August point-MAE selector narrowly preferred trial13, not trial23; do not switch that persisted overall choice because of September. Report quantile-cohort selection explicitly. Pure requested seven-day raw input did not reach5.2. Analog90 adds strictly prior90-day analog retrieval and historical interpolation-error channels; it is not raw-only or a blend with the older trained model.

Fixed16-filter2x8 paired Q/H encoder and small1x1 lag fusion; independent constrained point and ordered11-marginal-quantile heads. Raw4608total/3860point parameters; analog5696total/4948point. Only point conserves means, max error5.684e-14 EUR/MWh. Independent q50 is not claimed to equal the feasible MAE-optimal decision. Source protocol records exact inputs/losses/settings; source snapshots and hashes preserve training and repaired finalization versions.

## Uncertainty and interpretation
Analog quantile raw90% interval coverage91.1892%, mean width28.0394EUR/MWh. August-calibrated90% interval coverage92.3524%, width28.9927. Calibrated95% coverage96.1892%, width37.7092. These are quarter-level empirical coverages, not whole-day guarantees; whole-day interval inclusion was0/30 for these bands. Calendar dependence and16calibration days limit formal claims. Ordering and calibration were independently reconstructed.

Saved normalized kernels for all input channels, shared/head weights, signed/positive Grad-CAM, local activation-gradient maps, weight-randomization correlations and consistent day-removal ablations under results/quantile_2d_cnn_20261017/explanations/. Maps target a selected quarter correction and90% interval width, never the fixed daily sum. Examples are explicitly posthoc diagnostics: first September day, worst actual error, highest predicted severe-risk day. Qualitative maps do not establish causal explanations.

## Error confidence and sparse catastrophes
Primary predeclared event was joint four-zone dayMAE>10EUR/MWh. Only1 event among182 earlier held-out days and0 among30 September days. Therefore all four logistic feature/layer probes correctly used a constant fallback. No catastrophe detector was validated; small Brier scores here merely reflect event scarcity. Do not interpret null AP/AUC/recall as success.

Added explicitly exploratory continuous daily-MAE ridge probes, retaining the catastrophe definition and all reconstruction choices. Train on182 earlier out-of-sample errors, fixed ridge10; calibrate a regularized intercept on August16–31. All four feature sets reported without September selection. September error when predicting daily reconstruction MAE:
- input summaries: .845746 EUR/MWh, Spearman .410011.
- inputs+quantile/point summaries: .788694, Spearman .363737.
- plus conv summary: .957819, Spearman .193326.
- plus fusion summary: .887661, Spearman .248053.
Constant prediction from calibration days:1.008734. Raw90% interval width alone correlates with actual dailyMAE at Spearman .304116. This is a modest exploratory day-error signal; intermediate-layer summaries did not improve over the simpler inputs+uncertainty probe. Thirty days cannot establish rare-catastrophe detection. Coefficients, normalizers, calibration offsets and predictions independently replayed.

## Execution history and receipts
Initial Nightwatch74129e6183143d5346163a8b, worker5702, log/root/.local/share/nightwatch-hackaton/74129e6183143d5346163a8b.log. All training completed, then reporting crashed because a NumPy bool was not JSON-serializable. Fixed conversion to native bool; no training, hyperparameter or persisted selection changed. Original tracker correctly remains crashed.
Recovery Nightwatch2859d80fce9e2fbe5ef622a6, worker6216, log/root/.local/share/nightwatch-hackaton/2859d80fce9e2fbe5ef622a6.log completed, exit0; success/failure Telegram reporting verified through execution-host current Nightwatch wrapper and live API. Subsequent short continuous-confidence analysis and final audit also exited0. completion.json preserves both outcomes.

Remote outputs/workspace/HACKATON/results/quantile_2d_cnn_20261017. Artifact archive/workspace/HACKATON/results/quantile_2d_cnn_20261017_artifacts.tar.gz excludes only regenerable raw/analog feature caches; contains checkpoints, predictions, official-data snapshot, curves, metrics, plots, risk probes, explanations, source snapshots and audits. Local backup retrieval initiated in this turn; verify extraction/completeness before treating it as backed up.

## Earlier tuning completion verified
Earlier42fit tuning Nightwatch038fce95c7be0dda153059b8 completed its independent audit. Its full-August-selected analog90 trial35 had September5.177024160106511,4948parameters. All42trials, selected checkpoints/predictions and reports locally backed up at results/parallel_2d_tuning_20261017. This is a separate experiment with different validation allocation/normalization; do not claim a matched quantile improvement over it. It supersedes the prior9/42pending record.

## Prior startup summary preserved
# HACKATON project memory

Objective: official-only four-zone quarter-hour reconstruction; hourly means within 0.0001 EUR/MWh. Strictly earlier history, no external/pretrained/October prices. Train302 through July/August31 selection/September30 developmental evaluation, not blind. Recent inference history cannot authorize retraining/recalibration. Zones anonymized.

Current request 2026-10-17: restart design around simple interpretable2D CNN, direct quantiles, constrained MAE, Grad-CAM and catastrophic-day risk. User clarified24points means query-day24hourly prices. Pair N-day Q/H histories (H repeated4times), queryH known/queryQ zero, explicit mask. Proposed separate constrained-MAE and ordered-marginal-quantile heads; not all quantiles can conserve the same hourly mean. Risk requires chronological out-of-sample errors, predefined catastrophe threshold and independent evaluation; interval width alone is insufficient. Design only; no new campaign launched or prior job canceled. Next specify bounded matched loss comparison and risk event. Detailed design and verbatim previous startup: memory/records/20261017T114709-2d-cnn-quantile-restart-and-catastrophic-day-risk-design-202-b5884e22.md.

Previously verified2026-10-17: monthly_joint_blend_gpu1 SeptemberMAE5.091209; interpolation7.090321. Offline363-day refit audited, no submission (DEPLOYMENT_REFIT.md; results/COMPLETION_AUDIT.md). Audited parallel2D history5.431127/no-history5.903767,1388/940parameters (PARALLEL_2D_CNN_PROTOCOL.md; results/parallel_2d_zone_cnn_20261017). Prior linear-quantile design: LINEAR_QUANTILES_PROTOCOL.md.

Pending, not live-rechecked this turn:42fit tuning, targetMAE5.1–5.2, last9/42 observed; TUNED_PARALLEL_CNN_PROTOCOL.md; results/parallel_2d_tuning_20261017. VastRTX5090 root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python; Nightwatch038fce95c7be0dda153059b8,worker5203; log/root/.local/share/nightwatch-hackaton/038fce95c7be0dda153059b8.log. Retrieve final audit independently.

Other experiments/job pointers preserved in the design record and memory/records/20261017_startup_before_2d_tuning.md. User values simple filters and robustness. SSM untested.


## Next step
Present final audited results, explain the raw-versus-analog and overall-versus-quantile selection distinctions, and inspect saved kernels/activations as desired. No new model search is required to satisfy the approximate5.2 quantile target. Rare-error confidence remains unvalidated.

## Final local backup and visual verification
Artifact archive downloaded and extracted at results/quantile_2d_cnn_20261017/.
Remote/local archive SHA256 both eabafcb24de5b8d2a40e3b99795e079bd1a4226ed0237b830f2f5e945ec41c71.
Verified24main and6fold model.pt files, selected trial23 predictions, explanations,
final audit passed and continuous-confidence replay passed. The two large raw/analog
feature caches are regenerable and excluded; all scientific outputs and source
snapshots are backed up. The MAE chart and September1 signed attribution plots were
visually inspected. Standard Grad-CAM spreads activation across the day because
it averages spatial gradients; the local activation-gradient map concentrates at
the selected hour. Prefer the local signed map when interpreting this local-output
CNN; the standard map does not demonstrate causal influence across all times.
REPORT.md opened in Codex. Prior9/42pending managed record marked superseded;
the initial quantile design marked historical after implementation.
