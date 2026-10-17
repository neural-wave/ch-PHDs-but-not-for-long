---
title: User simplified joint CNN to one 16-channel hidden layer
status: current
kind: decision
recorded_at: '2026-10-17T10:27:23.846635+00:00'
sources:
- SIMPLE_JOINT_ZONE_CNN_PROTOCOL.md
- joint_zone_cnn.py
- run_simple_joint_zone_cnn.py
experiment_id: simple_joint_zone_cnn_20261017
supersedes: []
---

2026-10-17 user asked CNN as simple as possible. Fixed Conv1d(n,16,5),ReLU,Conv1d(16,4,1), no dilation/deep stack/search. Current hourly4 + quarter-position onehot4 channels; history adds seven raw96-quarter and repeated24-hour curves for four zones plus masks, total71channels. Exact parameters724 without history /5764 with. Only six fits: two history conditions x seeds17/43/71, fixedLR.001; August selects seed/epoch, no ensemble. Same train302/Aug31/Sept30 and conservation. New Nightwatch799366803bf6fbf860494899 worker4860, log/root/.local/share/nightwatch-hackaton/799366803bf6fbf860494899.log; root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python. Health/initial running heartbeat180s verified; ETA5min conservative. Earlier144fits completed training then audit failed at validation MAE replay; root cause training disabledTF32 while finalize/audit default enabledcuDNNTF32. Fixed module-level flags for all paths; replayed same saved checkpoints and unchanged August selection, independent audit passed. Broad-search September5.862788 without history,5.485715 with; separate from minimal experiment. Old Nightwatch54daabf3443590d977ed8fb5 retains actual original crash history, correction was synchronous verified replay. Archived startup memory/records/20261017_startup_before_minimal_joint_cnn.md.
