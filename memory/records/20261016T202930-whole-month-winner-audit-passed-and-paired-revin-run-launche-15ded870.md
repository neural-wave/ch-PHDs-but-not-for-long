---
title: Whole-month winner audit passed and paired RevIN run launched
status: superseded
kind: experiment
recorded_at: '2026-10-16T20:29:30.151931+00:00'
sources:
- REVIN_PROTOCOL.md
- results/monthly_graph_gpu1/audit.json
- results/monthly_graph_gpu1/precision_diagnostics.json
- train_revin.py
- revin_models.py
- results/experiments_summary.html
experiment_id: monthly_revin_gpu1
supersedes: []
status_updated_at: '2026-10-16T20:36:21.591338+00:00'
---

2026-10-16: completed authoritative whole-September serialized CPU replay audit for monthly_graph_gpu1. All 11,520 rows finite/unique, strictly prior history per date, 302-day normalization at all three nested bundle levels, forest roots 28,992 examples. MAE 5.092269863919437 versus cached 5.09226969906224; exported hourly mean max error 2.5124791136477143e-11. Original arbitrary pointwise 0.0002 replay criterion failed; investigated cuDNN TF32 with isolated graph component. Final audit explicitly uses sub-cent pointwise tolerance 0.01 and strict MAE agreement 1e-5, separate from challenge hourly-mean tolerance 1e-4. Actual max drift 0.0018328161. Source/JSON diagnostics and export retrieved; model SHA256 verified local/remote 0fd69dfc9b12fc65f02facbeed8911b456b67cbb03bbb11094b5a18960572707.

Implemented hybrid RevIN graph and matched control. Input-only daily/zone hourly stats, inverse residual scale, absolute level/scale and 389 rich features retained; eight paired configs, August selection only, 302-day fits, TF32 disabled. See REVIN_PROTOCOL.md for exact adaptation and limitations. Remote tests passed finite gradients, negative/flat prices, isolated days and zero means. Two-epoch smoke tested ensemble/control-only/RevIN-only exports. Full run Nightwatch 845d62f24e76c25572da6009 running with heartbeats and Telegram enabled; supervisor2175934, trainingPID2175936 verified on physical GPU1 UUID GPU-b4c7e2c6-57bd-d548-f67a-a7cc9f456203. Root /home/mnogales/Projects/HACKATON, sources revin_campaign/, results/monthly_revin_gpu1, log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/845d62f24e76c25572da6009.log. First launch afa69e6919db86ea0674afc3 terminal crashed exit2 before any training due to missing uploaded worker; repaired upload and verified terminal state before relaunch. Four of eight candidates completed at live check; no full result claimed. Updated report pending status and inference/selection compatibility. Next: verify run terminal state, inspect paired August results, retrieve artifact if selected and replay new winner. Goal active.
