---
title: Normalization audit and matched four-arm eight-month quantile experiment launched
status: superseded
kind: experiment
recorded_at: '2026-10-17T15:04:51.626328+00:00'
sources:
- ssm_experiment.py
- REVIN_PROTOCOL.md
- VALIDATION_PROTOCOL.md
- campaigns/normalization_20261017/PROTOCOL.md
- campaigns/normalization_20261017/train.py
- campaigns/normalization_20261017/audit.py
- campaigns/normalization_20261017/run.py
- results/normalization_20261017/launch.json
- results/normalization_20261017/smoke/audit.json
- memory/records/20261017_startup_before_normalization.md
experiment_id: normalization_20261017
supersedes: []
status_updated_at: '2026-10-17T15:18:36.457346+00:00'
---

User asked whether normalization was absent and requested testing its usefulness. Verified actual owning code: ssm_experiment.normalize (lines43ff) and current quantile campaigns already use fitting-set feature z-scores, scale floor .05, clipped [-15,15], with history-mask restoration; residual outputs have fixed scale20. Prior REVIN_PROTOCOL.md describes the completed 2026-10-16 eight-candidate September-only hybrid reversible-normalization comparison: RevIN won two/four matched settings, average August standalone delta -0.006642 EUR/MWh, ensemble September descriptive improvement CI spanning zero. That is historical limited evidence, not the current eight-fold ablation.

Implemented isolated campaigns/normalization_20261017. Four arms: existing fixed engineered units without fitted standardization; fitting-only z-scores; query-day/zone input normalization followed by fitting z-scores; same input normalization with reversible output scale. Statistics mu and sqrt(population variance+1) use only each zone's supplied24hour curve. Normalize price/slope/spread/prior-contrast features and all sequential residual/level history using that query-day zone scale. Preserve absolute mean/std side inputs, masks, clocks and feature dimensions. No learned affine parameters; adaptation rather than exact RevIN replication. Fixed retains engineered /100,/20 units and common clip. Each arm trains exactly the same original-currency MAE+pinball objective divided by fixed20; output reversal never introduces inverse-volatility day weighting.

Matched compact-summary CNN/S4D/S5 width32/state32; rates.001/.003; seeds17/43;48fits/fold,384fullfits;300epochs/patience35, unchanged optimizer/batch/clip. Eight expanding February–September tests, preceding month days1–15 select,16–end calibrate, fitting-only preprocessing, no refit on selection/calibration dates. Default11ordered quantiles and50/80/90/95CQR retained. q50 conserves hourly means. Report equal-month ratio, raw/pooled ratio, per-zone and alternate interpolation scores, pinball, intervals and width/error diagnostics. Pre-register paired candidate and validation-selected differences. No deployment update.

Preflight on existing authorized Vast RTX5090 passed: first/last-fold24two-epoch fits; independent normalization equations, heldout preprocessing invariance, flat/negative inputs, physical loss equivalence, all checkpoint replay (maximum difference0), quantile order, hourly conservation, chronology and recomputed CQR/ranks/scores. Parameter counts identical across arms: compact19180,S4D31020,S531036. Mac only edited source and checked syntax; all feature preparation/fitting/audit ran remotely.

Launched at17:03:21Zurich17October2026 with current copied Nightwatch CLI; run73994759a86ac27184894e2a, ETA17:18:21. Host root@93.91.156.85:55340,/workspace/HACKATON; Python/venv/main/bin/python; worker10712; log/root/.local/share/nightwatch-hackaton/73994759a86ac27184894e2a.log. Tracker verified running with expected180s heartbeats; a subsequent heartbeat and actual full fitting logs verified February completed/March underway. Health OK. Success/failure Telegram alerts authorized by existing Nightwatch workflow. Full success requires independent384checkpoint audit plus recovery archive, not ETA expiry. Local receipt results/normalization_20261017/launch.json and smoke/audit.json. Next return after notification: inspect actual state, retrieve REPORT/metrics/audit, compare paired normalization deltas, download and hash-verify completed archive.

Separately verified owning remote corrected SSM campaign now complete/audited160checkpoints (prior startup was stale): results/ssm_monthly_20261017/audit.json all checks passed. Remote summary tcn ratio.778941946,s4d.785553825,s5.788694891,compact.797603409; selector.787637818. Existing local completion.json records428504645byte archive SHA65910d1879d96f06817b907a6aed755ee33c256d218dc09eb3d14bca453136d1; local archive download was observed growing then reached matching size, not independently hashed in this turn. Do not claim backup verification solely from size. Prior complete startup preserved verbatim in memory/records/20261017_startup_before_normalization.md; prior quantile/zone-weight results remain unchanged.
