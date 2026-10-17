---
title: Eight-fold S4D S5 quantile comparison completed and audited
status: current
kind: experiment
recorded_at: '2026-10-17T15:03:21.643768+00:00'
sources:
- VALIDATION_PROTOCOL.md
- campaigns/ssm_monthly_20261017/PROTOCOL.md
- results/ssm_monthly_20261017/REPORT.md
- results/ssm_monthly_20261017/audit.json
- results/ssm_monthly_20261017/completion.json
experiment_id: null
supersedes:
- memory/records/20261017T145758-corrected-s4d-s5-comparison-to-required-eight-monthly-folds--569cca24.md
---

Corrected SSM comparison finished all160fits and eight February–September expanding folds. Nightwatch f1471cb72fbc1d8ade56973a completedexit0 at2026-10-17T14:58:44UTC(16:58Zurich). Full160checkpoint audit passed: chronology, fold-training-only statistics, all candidate checkpoint/validation replay, q50 conservation/order, independent calibration ranks/coverage/widths, equal-month ratio aggregation. Remote /workspace/HACKATON/results/ssm_monthly_20261017 on existing root@93.91.156.85:55340 RTX5090. Models/preprocessing fitted from scratch, previous-month days1–15 selection,16–end calibration, frozen tests, only prior observed quarters. All241testdays/92544quarters; developmental backtests.

Results(mean monthly ratio,pooledMAE,calibrated90%coverage,widthEUR): TCN0.778942,4.512701,90.16%,20.625; S4D0.785554,4.551869,88.62%,20.029; GRU0.785659,4.551947,89.08%,20.276; validation-selector0.787638,4.566457,90.31%,21.515; S5 0.788695,4.568278,89.23%,20.625; compact0.797603,4.629994,89.17%,21.067; interpolation1.0,5.814817,89.50%,27.784. Thus bothSSMs improve interpolation/compact control but do not beat matchedTCN. Below-nominal empirical SSM90%coverage is reported without a guaranteed-coverage claim. Eleven quantiles,50/80/90/95 calibratedbands, per-month/zone metrics, pinball and relative-width/error diagnostics saved. Full originalS4 not tested; S4D explicitly its diagonal variant. Feature/budget matched, parametercounts differ. No deployment change or outer-test tuning.

Original56fit pilot259bd2a9188ede4ad8ea466b remains separate exploratory evidence. SSM_PILOT_PROTOCOL.md annotated accordingly. New source is campaigns/ssm_monthly_20261017; existing quantile interval/report helper reused with independent SSM audit. Remote archive results/ssm_monthly_20261017_completed.tar.gz,428504645bytes,SHA25665910d1879d96f06817b907a6aed755ee33c256d218dc09eb3d14bca453136d1. Local transfer underway at record time; do not claim verified local backup until backup_verification.json exists. Completion receipt already local. Next complete local backup hash check/extract and inspect fuller month/quantile results as needed. Prior unrelated startup preserved in memory/records/20261017_startup_before_ssm_protocol_correction.md.
