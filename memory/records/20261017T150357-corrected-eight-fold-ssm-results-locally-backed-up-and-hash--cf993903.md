---
title: Corrected eight-fold SSM results locally backed up and hash verified
status: current
kind: backup
recorded_at: '2026-10-17T15:03:57.736415+00:00'
sources:
- results/ssm_monthly_20261017/backup_verification.json
- results/ssm_monthly_20261017/completion.json
- results/ssm_monthly_20261017/REPORT.md
- results/ssm_monthly_20261017/metrics.json
experiment_id: null
supersedes: []
---

Completed local recovery backup of corrected eight-fold campaign f1471cb72fbc1d8ade56973a. Archive results/ssm_monthly_20261017_completed.tar.gz exactly428504645bytes and SHA25665910d1879d96f06817b907a6aed755ee33c256d218dc09eb3d14bca453136d1 matches remote completion receipt. Extracted all results, checkpoints, raw eleven quantiles, calibrated intervals and92544-quarter keyed selector export to results/ssm_monthly_20261017. Local audit confirms8folds/160checkpointspassed. backup_verification.json and launch.json updated completedexit0. No active SSM training/download remains. Mean monthly pinball/interpolation-MAE normalized ratios: TCN.228798,S4D.233245,S5.235077. This pinball statistic uses the official point-baseline MAE denominator and is not a pinball-to-pinball ratio; per-quantile raw pinball values are in each fold metrics. Full ordinary point/quantile summary preserved in completed/audited study record20261017T150321-eight-fold-s4d-s5-quantile-comparison-completed-and-audited-c7afe5fe.md. BothSSMs improve interpolation but do not beat matchedTCN in this bounded developmental comparison; no statistical significance claim, no deployment replacement. Original56fit pilot remains exploratory separate evidence.
