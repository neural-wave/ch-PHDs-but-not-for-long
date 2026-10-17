---
title: 'Monthly-ratio minimization campaign launched: 384 paired neural fits and matched
  controls'
status: superseded
kind: launch
recorded_at: '2026-10-17T13:58:18.295641+00:00'
sources:
- campaigns/monthly_ratio_20261017/PROTOCOL.md
- campaigns/monthly_ratio_20261017/experiment.py
- campaigns/monthly_ratio_20261017/audit.py
- campaigns/monthly_ratio_20261017/run.py
- results/monthly_ratio_20261017/launch.json
- results/monthly_ratio_20261017/manifest.json
- results/monthly_ratio_20261017/smoke/audit.json
experiment_id: monthly_ratio_20261017
supersedes: []
status_updated_at: '2026-10-17T14:07:44.938275+00:00'
---

User authorized on17October2026: minimize average monthly ratio, try several models. Implemented isolated campaign targeting adopted equal-month MAE/corrected-interpolation-MAE objective, not pooled MAE alone. Paired ordinary versus month-balanced relative training: each fitting day weight proportional to1/(training-month day count × that training month's interpolation MAE), normalized to mean1. Training targets alone determine weights; preceding validation month's first15days stop/select; calibration16-end reserved; outer test never selects within a fold. Same8expanding test folds February–September,241days/92544targets. No full nested inner resampling is claimed.

Six gradient families: rich linear1170parameters; compact history linear585; compact additive2331; compact MLP7395; rich temporal CNN32483; global-hourly-conditioned analog90 small2DCNN16676. Each uses2learningrates(.001/.003),2seeds(17/43),both objectives:48fits/fold,384total. Add160ridgefits(2featurepacks ×2objectives ×5alphas ×8folds), analog shrinkage, interpolation, bounded global blends and validation-only recipe selector. Today's complete4×24hourly curve conditions every learned model. Strictly prior observed history; exact mean conservation. No new paid instance, pretrained/external inputs, submission change or local training.

Preflight encountered remote cached compact-feature API mismatch and NumPy2 string reductions. Corrected in the new isolated campaign; imported exact local dependency snapshot in campaigns/monthly_ratio_20261017/sources/ instead of overwriting shared remote modules. Local/remote manifest source hashes verified. Archived earlier preflight manifest on server; no full model fits existed before final source freeze.

Prelaunch first/last-fold smoke passed24neural fits,44candidate prediction rescoring,24serialized checkpoint replays exactly0error, maxhourlymeanerror1.1368683772161603e-13. Independent audit checked official source/baselines, split chronology, training-only weights/normalization/knots, family/blend/recipe selection and all scores. Smoke results are2-epoch runtime diagnostics, not full-study quality results.

Nightwatch b6d61339b6b5b82131b36c0d launched2026-10-17T13:57:31Z /15:57:31Zurich, estimated20minutes (16:17Zurich). Host root@93.91.156.85:55340; /workspace/HACKATON; /venv/main/bin/python campaigns/monthly_ratio_20261017/run.py. WorkerPID7626, log/root/.local/share/nightwatch-hackaton/b6d61339b6b5b82131b36c0d.log. Initial APIstatusrunning, heartbeat_expectedtrue/timeout180seconds, process/live log and repeated tracked smoke/audit verified; Telegramenabledtrue. Current CLI copied from installed source before launch. Detached wrapper handles success/crashes/signals; final success requires full independent audit and recoveryarchive results/monthly_ratio_20261017_completed.tar.gz. Full results pending at handoff. Existing benchmark0.738303meanmonthlyratio/4.273508pooledMAE. Next inspect actual Nightwatch state, REPORT.md/audit.json/summary.csv and retrieve checkpoint archive with hash before claiming outcomes.

## Verbatim previous startup

# HACKATON project memory

Objective: official four-zone quarter-hour reconstruction; hourly means within 0.0001 EUR/MWh. No external/pretrained components, October evaluation prices or inference-time refitting in submissions. Quarter history strictly prior. September developmental; zones anonymized; user prefers small interpretable CNNs.

Verified 2026-10-17: adopted VALIDATION_PROTOCOL.md: eight expanding chronological folds, three initial training months, next-month days1–15 selection/16–end calibration, following-month test (February–September). Fresh train-only fitting/normalization; frozen weights. Primary cross-month score: equal-month mean(MAE / same-month mean-corrected-linear-interpolation MAE); retain pooled raw MAE. Fixed-September history sizes separate. Single-block inner selection, not full nested selection.

Audited quick rescore: nine methods,241days/92,544targets. PooledMAE/monthlyratio: linear4.273508/0.738303; linearquantile4.278685/0.739333; additive4.366023/0.753150; ridge4.373205/0.754628; hierarchy4.419719/0.762695; direct4.433215/0.765104; CNN4.488442/0.773003; MLP4.476401/0.776817; interpolation5.814817/1.0. Official baselines independently recomputed. Evidence: results/normalized_validation_20261017/{REPORT.md,audit.json}; reproduce evaluate_normalized_validation.py. Full simple-backtest receipt:16folds/3,680candidates at root@93.91.156.85:55340,/workspace/HACKATON/results/expanding_simple_5090. No new training.

Prior September-only evidence: jointblend5.091209; analog90CNN5.177024; quantilecohort5.216651(overall selector5.336198); matched monthly estimates absent. Offline363-day refit audited/no submission. Exploratory FlowState completed/audited: control5.454209,FlowState+CNN5.461774; no incremental benefit/submission change. Archive results/flowstate_20261017_completed.tar.gz verified. Nightwatch102ec862033e194179693214 completed exit0; no active FlowState job. From-scratch SSM untested.

Detail and verbatim previous startup: memory/records/20261017T134724-monthly-chronological-validation-adopted-nine-method-interpo-5c71d2bf.md. Next apply adopted scoring before new tuning; newer CNN/blend matched-fold training remains unperformed.


Measured progress update at13:59UTC/15:59Zurich: full study4/8folds completed (50%), running with fresh heartbeat. Revised Nightwatch due_at1792245852.5897093 (16:04Zurich), about5minutes remaining including audit/archive. Full results still pending. Local launch.json retains the initial20minute estimate and this measured revision.
