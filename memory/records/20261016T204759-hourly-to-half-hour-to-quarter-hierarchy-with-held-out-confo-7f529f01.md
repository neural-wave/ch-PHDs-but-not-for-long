---
title: Hourly to half-hour to quarter hierarchy with held-out conformal launched on
  Vast
status: historical
kind: experiment
recorded_at: '2026-10-16T20:47:59.902016+00:00'
sources:
- HIERARCHICAL_PROTOCOL.md
- hierarchical_models.py
- train_hierarchical.py
- predict_hierarchical.py
- verify_hierarchical.py
- run_hierarchical_vast.py
- memory/records/20261016_before_hierarchical_startup.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-16T21:07:45.126870+00:00'
---

2026-10-16: User authorized a hierarchical hourly→30min→15min experiment on root@ssh1.vast.ai port24199 and asked to add conformal prediction. Implemented isolated mean-conserving sequential heads and matched direct encoder controls: widths32/64, seeds17/43, eight neural candidates plus interpolation, max220 epochs/patience30. No pretrained/external data. Train302 days Oct–July; August1–15 stopping/model selection, August16–31 held exclusively for calibration, September30-day development test. Previous monthly selection and Ando deployment refit are preserved; previous startup summary copied verbatim to memory/records/20261016_before_hierarchical_startup.md. Historical Ando job status not rechecked in this conversation.

Conformal finite-sample rank ceil((n+1)*(1-alpha)); 90/95% zone/quarter marginal and zone/hour joint intervals, 90% day-joint intervals from16 day blocks. 95% day-joint would require infinity and is omitted explicitly. Temporal dependence/drift/history use mean nominal coverage is not guaranteed; empirical September coverage and widths required. Point hierarchy means are exact; interval endpoints are envelopes and need not preserve means. Calibration never chooses candidate/epoch/weight. All-August scores descriptive, not an equal-selection-budget claim versus historical winner.

Remote RTX PRO4000 Blackwell 24467MiB was idle at setup. Read full /etc/vast-agents-guide.md. Runtime /venv/main/bin/python (Python3.12.15), PyTorch2.8.0 CUDA12.8, NumPy2.4.6/pandas2.3.3 installed. Official data/source transferred to /workspace/HACKATON. CPU local syntax checks passed. Remote GPU preflight passed nonzero hierarchy conservation, finite gradients, corrected conformal ranks including infinity, and causal-feature mutation test.

Initial Nightwatch57114bfaba49bf1217220b0b crashed before training due to numpy.str_ in synthetic fixture; corrected to object dtype and reran preflight successfully. Restart Nightwatch5c9227f73977427b05cb824b verified running, heartbeat_expected true,180s timeout, worker1380, ETA35m from22:47:28 Zurich (estimated23:22). Log /root/.local/share/nightwatch-hackaton/5c9227f73977427b05cb824b.log. Current Nightwatch CLI installed on execution host; private config mode600 /root/.config/nightwatch/client.json; completion/crash/heartbeat notifications authorized. This job includes setup idempotent checks, two-candidate smoke, full campaign, saved model replay and offline point/interval CSV audits. Outputs /workspace/HACKATON/results/hierarchical_vast/{metrics.json,test_predictions.npz,test_predictions.csv,candidate_*.pt}; smoke results in results/hierarchical_smoke_vast. No accuracy/coverage result yet. Next verify actual terminal status, retrieve artifacts, compare hierarchy/direct/interpolation, review coverage/widths, and decide whether to extend. Do not infer success from ETA.
