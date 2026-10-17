---
title: New RTX 5090 simple architecture search launched
status: superseded
kind: experiment
recorded_at: '2026-10-17T08:26:33.862571+00:00'
sources:
- simple_search.py
- run_simple_search.py
- SIMPLE_SEARCH_PROTOCOL.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T08:29:43.616168+00:00'
---

2026-10-17: user authorized new GPU root@93.91.156.85:55340 and maximal parallel architecture search. Interpreted last month as September evaluation (stated to user), consistent with existing monthly protocol. Instance guide read; idle RTX5090 32GB UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9; CUDA matmul preflight passed. Dedicated /workspace/HACKATON, new runtime installed (torch2.14.1+cu130). Existing remote jobs untouched.

224 single models across compact/rich features, flat/interpolation anchors, linear/additive/shallow MLP/tiny temporal CNN families; learning rates .001/.003, seeds17/43. Training302 days throughJuly; August31-day selection, September diagnostic only (already developmental). Train-only normalization/hinge knots, official source hash verified, causal first-test target perturbation checked. Full trials enforce exact hourly means, finite predictions, serialized replay; best models frozen by August before September scoring. No external/pretrained/October data, no ensemble. Exact parameter counts saved.

Remote source and outputs /workspace/HACKATON/results/simple_5090. Nightwatch2093a7ebed349785229d04bf, worker1583, runner1584; log /root/.local/share/nightwatch-hackaton/2093a7ebed349785229d04bf.log. Heartbeat expected:true,180-second timeout, success/crash notification enabled. Started08:24:53UTC (10:24:53Zurich); initial45-minute estimate conservative.

Smoke16 jobs at five epochs: 4workers8.0606s,8workers6.1541s,12workers6.8449s. Selected8 workers by measured short-workload wall time (not a full-run speedup claim). Full process1769 observed157/224 trials done; GPU99%,5933MiB,152.62W. No completion claim yet. Next verify terminal state, copy result/model artifacts locally, visually inspect PNG/SVG, report validation-selected family metrics and complexity. Earlier startup context archived in memory/records/20261017T082550-startup-summary-archived-before-new-5090-simple-search-ec074e37.md.
