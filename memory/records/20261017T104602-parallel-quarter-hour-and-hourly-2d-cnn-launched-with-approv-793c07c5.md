---
title: Parallel quarter-hour and hourly 2D CNN launched with approved late information
status: historical
kind: experiment
recorded_at: '2026-10-17T10:46:02.323678+00:00'
sources:
- PARALLEL_2D_CNN_PROTOCOL.md
- parallel_zone_cnn.py
- run_parallel_zone_cnn.py
- joint_zone_cnn.py
experiment_id: parallel_2d_zone_cnn_20261017
supersedes: []
status_updated_at: '2026-10-17T11:27:41.634315+00:00'
---

2026-10-17 user explicitly clarified quarter/hour must be parallel for2D kernels; approved current hourly price, neighboring-hour changes, quarter-position and weekday at final stage. Implemented four zone channels+availability mask; pairedQ/H rows for day-7..0, hourly24repeat4 to96, query-day quarter row always zero/masked. Grid5x16x96 (history) or5x2x96 (none). Conv2d5->8 kernel2x8 stride2x1 reads representations jointly, same96time positions (pad3left4right). ReLU, flatten eight day identities as channels, append23late channels, pointwise fusion8/ReLU/output4. Exact940/1388 parameters without/with history. Kernel spansQ/H, day mixing is in fusion head. Six fits only: historiesoff/on x seeds17/43/71, fixedLR.001, August selects; train302/Aug31/Sept30development; interpolationanchor, exact hourlyconservation; no U-Net/classification/deployment. Export selected normalized kernels perzone/mask, rawfilter+fusionweights and August1activations. RemoteidleRTX5090 verified, root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python. Nightwatchhealthpassed/currentCLIcopied, run1b9cf34ee35a62b780645756 worker5024, log/root/.local/share/nightwatch-hackaton/1b9cf34ee35a62b780645756.log, initialETA5min, heartbeat180s true. Wrapper prepare/smoke/full/audit/filterexport; initialrunning confirmed, numericalresults pending. Prior minimal1D completed/audited historySeptember5.539996 vsnone5.921824; local fullbackup verified. Prior startup preserved memory/records/20261017_startup_before_parallel_2d_cnn.md.
