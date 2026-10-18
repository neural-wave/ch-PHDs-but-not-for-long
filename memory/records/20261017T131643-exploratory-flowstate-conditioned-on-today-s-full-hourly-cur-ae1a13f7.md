---
title: Exploratory FlowState conditioned on today's full hourly curve launched with
  matched CNN control
status: superseded
kind: experiment
recorded_at: '2026-10-17T13:16:43.800741+00:00'
sources:
- campaigns/flowstate_20261017/PROTOCOL.md
- results/flowstate_20261017/launch.json
- results/flowstate_20261017/preflight.json
- results/flowstate_20261017/model_receipt.json
experiment_id: flowstate_20261017
supersedes: []
status_updated_at: '2026-10-17T13:19:14.061970+00:00'
---

## User-authorized exploratory FlowState comparison — 2026-10-17

User asked to try FlowState even though pretrained components are disallowed for submission, to measure reconstruction error. They explicitly corrected a projected-forecast proposal: today's hourly curve must condition the within-hour shape, not merely set hourly means.

Implemented campaigns/flowstate_20261017/{experiment.py,audit.py,run.py,preflight.py,PROTOCOL.md}. Both 9,156-parameter CNN arms receive all four zones' entire current 24-hour curve through a global Linear96->16 path, local hourly levels/slopes, quarter/weekday indicators and seven prior calendar days. FlowState arm adds eight frozen forecast/shape channels; same-capacity control zeros these channels after normalization. Exact hourly conservation through centered corrections around interpolation. FlowState r1.1:18,524,928 frozen parameters, checkpoint940580f969036cfca514cdff725f7b02c03e1bcf, IBM source640f7c80a21a4e2253b6d6140d067452bb93e7d4, scale.25/horizon96/28calendar days; missing DST days preserved as NaNs. Pretraining overlap not audited. Only official CSV used as task data; no October2026 prices or deployment changes.

Train indices28:302, August302:333 selects epochs/seeds, September333:363 developmental. Six fits, paired seeds17/43/71, max300epochs/patience30. Selections persisted before September FlowState inference. Includes projected-only baseline and independent scoring/replay/causality/whole-curve-sensitivity audit.

Verified preflight actual GPU inference4x96 finite, no missing/unexpected checkpoint weights, causal context/head inputs, nonzero distant-hour/cross-zone and FlowState gradients, two optimizer steps. Isolated venv inherits GPU torch2.14.1+cu130; transformers4.50.3/huggingface-hub.30.2/datasets3.5.1/accelerate1.6.0. Existing main environment untouched.

Launched Nightwatch102ec862033e194179693214 on root@93.91.156.85:55340, /workspace/HACKATON, worker7144; log/root/.local/share/nightwatch-hackaton/102ec862033e194179693214.log. Start13:16:19UTC, initial ETA15minutes; heartbeats expected/180seconds, actual success/crash Telegram alerts. Verified live progress208/305 frozen-feature dates at13:16UTC. No performance result yet; audit required for completion. Outputs/work receipt under results/flowstate_20261017. Next inspect actual terminal state and audited MAE, then back up completed outputs locally.

## Previous startup summary, preserved verbatim

# HACKATON project memory

Objective: official-only four-zone quarter-hour reconstruction; hourly means within 0.0001 EUR/MWh. Strictly prior history; no external/pretrained/October evaluation prices or inference-time refitting. September is developmental. Zones anonymized. User values simplicity and interpretability; approximately 5.2 MAE is acceptable.

Verified 2026-10-17: S4D/S5 ordered-history pilot launched on existing Vast RTX5090, root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python. Includes GRU/TCN and compact-summary controls, 36 August-selected trials plus 20 earlier fixed-config fits. Numerical recurrence, finite-gradient, stability and causal-input smoke checks passed. Last check: 8/36 initial trials complete; training active, no final scores/audit. Nightwatch 259bd2a9188ede4ad8ea466b, worker6908, log /root/.local/share/nightwatch-hackaton/259bd2a9188ede4ad8ea466b.log; subsequent heartbeat verified. Protocol: SSM_PILOT_PROTOCOL.md; local receipt results/ssm_pilot_20261017/launch.json. Next retrieve/back up results, inspect final audit.json and comparisons. No deployment change.

Prior verified: quantile CNN 24 fits/6 earlier folds audited; analog90 quantile-cohort winner September MAE5.216651,5696parameters, calibrated90% coverage92.35%. Overall August selector instead chose MAE-only trial13, September5.336198. User decision2026-10-17: set aside the unvalidated confidence/catastrophic-error detection branch unless reopened; preserve reconstruction results. Decision: memory/records/20261017T131629-confidence-and-catastrophic-error-detection-set-aside-by-use-4a35c213.md. Evidence results/quantile_2d_cnn_20261017; checkpoints locally backed up. Earlier analog90 tuning5.177024; monthly blend5.091209; interpolation7.090321. Offline363-day refit audited, not submitted.

Original startup and prior job/evidence pointers preserved verbatim in memory/records/20261017_startup_before_ssm_launch.md. Detailed quantile context: memory/records/20261017T120457-quantile-2d-cnn-completed-audited-mae5-216651-and-confidence-27bc0a19.md. SSM results remain pending; nonlinear SSMs are not intrinsically interpretable.
