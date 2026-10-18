---
title: Corrected S4D/S5 comparison to required eight monthly folds and quantiles
status: historical
kind: experiment
recorded_at: '2026-10-17T14:57:58.490154+00:00'
sources:
- VALIDATION_PROTOCOL.md
- campaigns/ssm_monthly_20261017/PROTOCOL.md
- campaigns/ssm_monthly_20261017/train.py
- campaigns/ssm_monthly_20261017/audit.py
- campaigns/ssm_monthly_20261017/run.py
- results/ssm_monthly_20261017/launch.json
experiment_id: null
supersedes:
- memory/records/20261017T131526-s4d-s5-56-fit-pilot-launched-on-existing-vast-rtx5090-46d37eac.md
status_updated_at: '2026-10-17T15:03:21.636473+00:00'
---

User corrected the validation method: eight chronological expanding folds February–September2026, fitting from October2025 through two months before test; preceding-month days1–15 select,16–end calibrate. Models/preprocessing fitted from scratch on fitting days only, frozen during tests, strictly prior observed quarters. Primary score arithmetic mean of eight monthly median MAE/mean-corrected-interpolation MAE ratios. Pooled MAE and quantile/calibrated coverage required. Read current VALIDATION_PROTOCOL.md: quantiles default eleven levels, CQR50/80/90/95,92544quarters; developmental backtests.

Acknowledged mismatch plainly. Old pilot259bd2a9188ede4ad8ea466b had already completedexit0 before correction,56fits and old audit; preserve as exploratory, not eight-fold evidence. Nothing was running to stop. Implemented isolated replacement campaigns/ssm_monthly_20261017; no old checkpoints or normalizers reused. Models S4D(diagonal variant),S5,GRU,causalTCN,compact-summary control. Width32/state32,LR.001/.003,seeds17/43;20candidates per fold,160fits. Ordinary conserved-q50MAE+mean pinball training;300epochs/patience35. Same168hour ordered history and compact query inputs; train-only uncertainty initialization. Existing audited quantile evaluation helper reused for calibrated intervals/confidence reporting; independent audit replays all candidate checkpoints, choices, calibration ranks, empiricalcoverage/width and scoring. Per-month/zone ratios, alternate uncorrected denominator and distinct pooled ratio saved. No deployment change, no catastrophe classifier.

Launched existing RTX5090 root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python under current Nightwatch. GPU verified idle before launch. New run f1471cb72fbc1d8ade56973a; worker10128; log /root/.local/share/nightwatch-hackaton/f1471cb72fbc1d8ade56973a.log.20-minute registered estimate. Heartbeat expected true timeout180sec, subsequent heartbeat verified. First/last-fold smoke audit passed10checkpoints over2folds. By last live check full training reached May fold; still running and no final full audit claimed. Final wrapper requires full audit and hashed recovery archive. Results remote results/ssm_monthly_20261017, local launch receipt and smoke audit. Next retrieve/back up and inspect final audit/REPORT when actual completion arrives. Old pilot managed record marked historical; do not confuse its completed status with replacement campaign.
