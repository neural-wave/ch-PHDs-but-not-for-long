---
title: Joint four-zone CNN with seven-day hourly and quarter history launched
status: historical
kind: experiment
recorded_at: '2026-10-17T10:24:15.733793+00:00'
sources:
- JOINT_ZONE_CNN_PROTOCOL.md
- joint_zone_cnn.py
- audit_joint_zone_cnn.py
- run_joint_zone_cnn.py
experiment_id: joint_zone_cnn_20261017
supersedes: []
status_updated_at: '2026-10-17T10:27:23.863204+00:00'
---

2026-10-17: user confirmed joint four-zone output (4x96), current hourly input (4x24), previous seven calendar days of both observed 96-quarter and 24-hour curves; requested matched with/without-history comparison and encouraged broader compute. Implemented 144 fits: history off/on, widths32/64/128, depths2/4/6, kernels3/5, LR .001/.003, seeds17/43. Joint channel-mixing dilated Conv1d over 96 quarters, interpolation anchor, exact hourly residual projection. Train302/August31/September30; September developmental only. Training-only normalization and no external/pretrained data; no deployment change. Smoke and final independent audit are mandatory wrapper stages. Existing idle RTX5090 live verified, root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python torch2.14.1+cu130. Current Nightwatch CLI copied and health passed. Registered run54daabf3443590d977ed8fb5, worker4472; log/root/.local/share/nightwatch-hackaton/54daabf3443590d977ed8fb5.log. Initial ETA45min, heartbeat_expected true timeout180s; actual completion/crash reporting enabled. Initial running launch confirmed, full numerical results pending. Outputs results/joint_zone_cnn_20261017. Next verify smoke/concurrency entry and later retrieve final audited outputs. Prior startup preserved verbatim memory/records/20261017_startup_before_joint_zone_cnn.md.
