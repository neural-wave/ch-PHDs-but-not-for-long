---
title: Champion submission refit launched with 348-day fit, 15-day held-out calibration
  and specification card
status: current
kind: deployment
recorded_at: '2026-10-17T19:50:47.872718+00:00'
sources:
- campaigns/submission_champion_20261017/recipe.json
- campaigns/submission_champion_20261017/refit.py
- campaigns/submission_champion_20261017/runtime.py
- campaigns/submission_champion_20261017/SPECIFICATION_CARD.md
- docs/SUBMISSION_REFIT_PLAN.md
experiment_id: null
supersedes: []
---

User requested a submission-ready champion and specification card, then explicitly approved reserving September 16–30 for calibration. This supersedes the prior all-363-days fitting preference. Fresh fitting uses 348 official dates, October 1 2025–September 15 2026. Freeze latest September-fold equal MLP/temporal-CNN/analog-2D-CNN blend, two seeds each, six networks with total weight 1/6 each. Epoch budgets: MLP17=112,43=134; temporalCNN17=54,43=33; analogCNN17=199,43=176. Calibration labels do not select epochs/settings or fit weights. Existing feature sources and official SHA256 frozen in recipe.json. Causal perturbation/truncation probes and exact cache checksum precede fitting. Final job includes separate held-out widen-only CQR, standalone CPU runtime replay, fresh CPU dependency installation, CSV contract/hourly conservation audit, specification card, manifest and archive. Coverage on those calibration dates is not future held-out validation. Research metrics remain development evidence, not refit measurements.

Verified remote host root@93.91.156.85:55340 reachable, RTX5090 initially idle, 1.9GiB disk available. Current Nightwatch wrapper copied before launch; tracker health passed. Nightwatch job 2b509aeec0a930d6a9979619 registered running, heartbeat_expected=true, timeout180s, workerPID16726. Command /venv/main/bin/python run.py campaigns/submission_champion_20261017/refit.py in /workspace/HACKATON. Log /root/.local/share/nightwatch-hackaton/2b509aeec0a930d6a9979619.log. Actual training confirmed: both MLP components complete at initial check, subsequent work ongoing. ETA35m is estimate only. No completion claim. Generated remote bundle /workspace/HACKATON/results/submission_champion_20261017/bundle/, audit.json, completion.json and champion_submission.tar.gz. Next: inspect real completion/audit, recover archive and verify checksum, deliberately replace deployment only after verified package. Historical submission/ and FREEZE_MANIFEST.json unchanged. Other unrelated working-tree changes preserved. Local syntax, weights and source hashes verified; source import passed on remote GPU environment. Fresh CPU dependency pins resolved in dry-run.
