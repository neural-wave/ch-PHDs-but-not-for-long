---
title: Launched trained RevIN quantiles and clarified chainsaw uncertainty diagnostic
status: current
kind: experiment
recorded_at: '2026-10-17T16:52:56.920275+00:00'
sources:
- campaigns/revin_quantiles_20261017/PROTOCOL.md
- campaigns/revin_quantiles_20261017/chainsaw.py
- results/revin_quantiles_20261017/launch.json
- results/revin_chainsaw_20261017/launch.json
experiment_id: null
supersedes: []
---

User explicitly chose TRAIN a RevIN quantile model, then clarified intended behavior: wider quantiles for predicted chainsaw quarter patterns and narrower for predicted smooth patterns; prior raw-drop examples did not test that. Implemented campaigns/revin_quantiles_20261017 with matched compact CNN fixed/day_reversible learned-positive-affine arms,2lr2seed8fold=64fresh fits. Original-currency loss,11quantiles, prior selection/calibration, hourly conservation, full checkpoint/CQR audit. Remote edge algebra/gradient/flat-negative/day-isolation checks and first/last4checkpoint smoke audit passed. Full job Nightwatch1e1315765ed2bc28e68bec9c worker14336 running (4/8folds at last observed progress). Host root@93.91.156.85:55340, project/workspace/HACKATON, results/workspace/HACKATON/results/revin_quantiles_20261017; log/root/.local/share/nightwatch-hackaton/1e1315765ed2bc28e68bec9c.log. Health verified, GPU idle before launch; success/crash owner Telegram enabled. Follow-up job04c6557a1e424750f3628707 worker14575 waits successful completion receipt then runs chainsaw.py; log same directory with follow-up ID, results/workspace/HACKATON/results/revin_chainsaw_20261017. Within-hour roughness=mean abs second difference(q50)/input daily scale. Shape cutoffs from prior-calibration roughness; compare normalized raw/CQR widths, observed roughness/errors and within-day24hour rank correlation; paired smooth/jagged hours in same dayzone selected using predicted shape only. Script separate from frozen trial manifest. Marginal uncertainty need not grow with predicted jaggedness; no relationship forced or result claimed. Launch receipts local results/<study>/launch.json; previous startup preserved verbatim memory/records/20261017_startup_before_revin_trial.md. Both ETAs10min, estimates only. No deployment changes.
