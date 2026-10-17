---
title: Linear compact RevIN preflight control-path fix and restart
status: current
kind: experiment
recorded_at: '2026-10-17T18:13:03.777058+00:00'
sources:
- campaigns/linear_compact_revin_20261017/train.py
- results/linear_compact_revin_20261017/launch.json
experiment_id: null
supersedes: []
---

First job94e67d8959b330d22189027a crashed during smoke after one fitted checkpoint: original control recipe fixed_linear_compact_ratio is a selection alias, not a trials-directory name. Fixed train.original_predictions(month) to resolve selected config.id via SOURCE/folds/month/selection.json and audit same owning lookup. No scientific settings changed. Preserved failed_preflight_manifest.json remotely; reruns all smoke/audit. Restart Nightwatch b451591368201a2ccd5551eb worker15718 log/root/.local/share/nightwatch-hackaton/b451591368201a2ccd5551eb.log, same host/project/results and10minETA. Launch receipt/startup updated; success/crash alerts active. Await launch verification and full checks; no completion claimed.
