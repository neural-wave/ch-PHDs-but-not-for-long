---
title: Monthly chronological validation adopted; nine-method interpolation-normalized
  rescore
status: current
kind: result
recorded_at: '2026-10-17T13:47:24.908842+00:00'
sources:
- VALIDATION_PROTOCOL.md
- evaluate_normalized_validation.py
- results/normalized_validation_20261017/REPORT.md
- results/normalized_validation_20261017/audit.json
- results/expanding_simple_5090/independent_audit.json
experiment_id: null
supersedes: []
---

17 October 2026: User requested formal cross-validation-like validation and month-difficulty normalization using MAE/interpolation_MAE. Adopted VALIDATION_PROTOCOL.md: eight expanding folds from Oct-Dec2025 train / Jan2026 selection-calibration / Feb test through Oct-Jul train / Aug selection-calibration / Sep test. Selection days1-15, calibration16-end, fresh train-only fitting/normalization, frozen weights, causal observed history, hourly conservation. Single chronological inner selection block, not full nested rolling selection. Primary cross-month metric is equal-month arithmetic mean of model MAE / same-month mean-corrected-linear-interpolation MAE. Retain target-pooled raw MAE, per-month/per-zone/worst-month ratios. Raw uncorrected interpolation denominators explicitly separate. Fixed-September history lengths excluded from CV aggregation; inspected months remain developmental.

Quick rescore complete without training/deployment changes. Previously completed full simple backtest CSV/metrics/audit retrieved from root@93.91.156.85:55340:/workspace/HACKATON/results/expanding_simple_5090. Receipt passed16folds/3680candidate evaluations/maxmeanerror1.1368683772161603e-13. Its six families plus existing direct/hierarchical/linearquantile campaigns give72monthly rows over8identical test months/241days/92544targets. Independently recomputed raw/corrected interpolation from official source SHA256 f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c; checked source/fold identity, CSV/report scores, simple-family validation argmins, daily/per-zone averages and existing independent audits. No full new prediction replay.

PooledMAE/meanmonthlyratio: linear4.273508/0.738303; linearquantilemedian4.278685/0.739333; additive4.366023/0.753150; ridge4.373205/0.754628; hierarchical4.419719/0.762695; direct4.433215/0.765104; CNN4.488442/0.773003; MLP4.476401/0.776817; interpolation5.814817/1.0. Every learned method beats interpolation all8months. Baseline3.848893Feb–7.090321Sep. Linear26.17% mean-monthly improvement; close quantile is descriptive, no significance/equal-budget architecture claim. Quantile selects joint MAE/pinball objective.

Separate September-only comparison: jointblend5.091209/0.718051(full-Aug selection); prior analog90CNN5.177024/0.730154; quantilecohort analog90_1 5.216651/0.735743; overall MAE-selected analog90_0 5.336198/0.752603. These lack eight-month estimates here. No new compute job launched. Apply adopted protocol to future campaigns; newer CNN/blend matched-fold training remains unperformed.

## Verbatim previous startup summary

# HACKATON project memory

Objective: four-zone quarter-hour reconstruction; hourly means within 0.0001 EUR/MWh. Submission requires official-only data, no pretrained/external components, October evaluation prices or inference-time refitting. History must be strictly prior. September is developmental, zones anonymized; user prefers small interpretable CNNs.

Verified 2026-10-17: separately authorized exploratory FlowState completed and audited. User required today's full hourly curve to condition shapes. Matched 9,156-parameter CNNs consume all 4x24 hourly prices globally, local slopes and seven-day history; one adds frozen FlowState forecasts. September MAE: FlowState+CNN 5.461774, control 5.454209, projected-only FlowState 7.109678; no clear incremental benefit. Hourly conservation, causal inputs, train-only normalization, August selection and replay passed. Pretraining overlap unaudited; no submission change.

Evidence: campaigns/flowstate_20261017/PROTOCOL.md; results/flowstate_20261017/{REPORT.md,audit.json,selection.json,model_receipt.json}. Complete local checkpoint/forecast archive results/flowstate_20261017_completed.tar.gz verified against remote SHA256. Nightwatch 102ec862033e194179693214 completed exit0 at 13:17 UTC/15:17 Zurich. Host root@93.91.156.85:55340, /workspace/HACKATON; log /root/.local/share/nightwatch-hackaton/102ec862033e194179693214.log. No active FlowState job. Next discuss findings; avoid September-driven tuning.

Prior audited/backed-up results: quantile-cohort analog90 MAE5.216651/5696parameters; overall August selector chose MAE-only5.336198. Earlier analog90=5.177024; monthly blend5.091209; interpolation7.090321. Quantile calibrated90% coverage92.35%, width28.99; catastrophe detector unvalidated. Quantile evidence: results/quantile_2d_cnn_20261017/{REPORT.md,audit.json,confidence_metrics.json,explanations/}; recovery2859d80fce9e2fbe5ef622a6 completed. Offline363-day refit audited/no submission: DEPLOYMENT_REFIT.md; results/COMPLETION_AUDIT.md.

Details: memory/records/20261017T131914-exploratory-flowstate-with-current-hourly-conditioning-compl-eeff6891.md. Verbatim previous startup remains in linked superseded launch record. Earlier quantile detail: memory/records/20261017T120457-quantile-2d-cnn-completed-audited-mae5-216651-and-confidence-27bc0a19.md. From-scratch SSM untested.
