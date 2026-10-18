---
title: S4D/S5 56-fit pilot launched on existing Vast RTX5090
status: historical
kind: experiment
recorded_at: '2026-10-17T13:15:26.608340+00:00'
sources:
- SSM_PILOT_PROTOCOL.md
- ssm_models.py
- ssm_experiment.py
- audit_ssm_experiment.py
- run_ssm_experiment.py
- results/ssm_pilot_20261017/launch.json
- memory/records/20261017_startup_before_ssm_launch.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T14:57:58.485728+00:00'
---

User explicitly authorized trying S4/S5 on existing Vast GPUs. Implemented bounded task-specific PyTorch S4D/S5 comparison plus GRU/TCN and compact-summary control. 168 historical hourly tokens across four zones, signed/absolute interpolation residuals, hourly levels, masks and clock/lag inputs. Same compact query features and mean-conserving decoder. 36 August-selected trials plus20 earlier fixed-config chronological fits;56 total. S4D is the diagonal variant, not full original S4. S5 uses HiPPO-normal diagonal conjugate-half initialization and affine parallel scan. Parameter counts differ and are reported; no claim of exact parameter matching. No deployment changed.

Remote root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python, RTX5090 UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9. Live preflight found GPU idle, CUDA working, 24GiB disk free, Nightwatch healthy. Read remote /etc/vast-agents-guide.md and local Nightwatch AGENT_GUIDE.md. Copied current installed Nightwatch wrapper to host.

Smoke passed: double-precision S4D FFT-vs-recurrence max error2.44249e-15; S5 parallel-scan-vs-recurrence6.66134e-16; finite gradients, stable discretized modes, all-family train-step/output checks. Data source hash and fixed-hourly current/future target perturbation passed. Local Python syntax compilation passed using /private/tmp cache; substantial computation remote only.

Nightwatch259bd2a9188ede4ad8ea466b launched2026-10-17T13:14:31UTC (15:14Zurich); worker6908, wrapper6909, training6931. Log /root/.local/share/nightwatch-hackaton/259bd2a9188ede4ad8ea466b.log. Registered30-minute estimate; heartbeat_expected true, timeout180s; dashboard/Telegram success/crash alerts through authorized workflow. Verified subsequent heartbeat at13:15:01UTC and running state;8/36 initial trials completed at last check, no final September score or audit yet. Do not call complete. Results remote results/ssm_pilot_20261017. Local launch.json preserves identity. Success requires audit_ssm_experiment.py to verify56 checkpoints, train-only normalization, calendar history, causal perturbation, August selection, recurrence smoke and hourly conservation. Later chronological fixed-config fits use earlier15-day epoch selection, not August-selected hyperparameters. All historical evaluation remains developmental.

Next: when notified, retrieve/back up complete results/checkpoints, verify final audit.json and report fair family/rolling comparisons. Previous startup summary preserved verbatim in memory/records/20261017_startup_before_ssm_launch.md; prior quantile/analog/deployment context remains there and in linked records.
