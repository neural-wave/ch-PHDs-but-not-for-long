---
title: Two full searches audited and retrieved; zone-specific refinement launched
status: superseded
kind: experiment
recorded_at: '2026-10-16T19:22:49.672112+00:00'
sources:
- results/LEARNED_MODELS.md
- results/search_gpu1/metrics.json
- results/rich_search_gpu1/metrics.json
- results/best_search.json
- results/artifact_sha256.txt
- results/zone_launch.json
- SEARCH_METHOD.md
- train_search.py
- predict.py
- run_zone_search.py
experiment_id: null
supersedes:
- memory/records/20261016T185131-gpu1-mae-search-implemented-smoke-verified-full-campaign-run-320ce740.md
status_updated_at: '2026-10-16T19:31:12.020697+00:00'
---

Previous goal turn made implementation/launch progress. Live checks this turn confirmed first run still training, then both first and rich campaigns completed successfully with exit0 and exported offline QA. Nightwatch d663a92035ce2527ea276864 and6f9db953fa9c7efa8259fe07 are completed, not active. Both have heartbeat/crash/completion alerts. Rich run staged separately in /home/mnogales/Projects/HACKATON/rich_campaign to avoid changing live original imports. Rich features passed192-row causality/current-future target poisoning/missing-history tests,307basic vs389rich features.

Measured MAEs: corrected interpolation validation5.5721848687/holdout6.6825855306; full basic ensemble3.7537877376/5.0361999622; rich3.6760608911/4.7872780226. Rich holdout improves28.36% vs interpolation and4.94% vs basic. Chosen by validation only. Smoke holdout was viewed; all later reported holdout scores are development diagnostics, not secret-test estimates. Historical targets only enter features on strictly later dates. Final artifacts refitted on all363days have no independent measured test result. Both final pipelines passed384-row exported prediction checks; hourly error~1.14e-13.

Downloaded metrics, heldout arrays, full models to local results/search_gpu1 and results/rich_search_gpu1. SHA256 matched remote/local: basic07132aa51f45be49df46cb861af209ef7bf63559481587cf38428c0401441624; rich a955b6b5fce9b0ee858f4dd5fb1b65365ffd5be95d9f0180599c44194d10cf7d. Code snapshots campaigns/basic and campaigns/rich. Current best results/best_search.json selects richer artifact,~326MB.

Next refinement adds validation-selected convex weights per zone, with fixed0.05 squared shrinkage towards shared weights. Same features,candidates and chronological split; no holdout selection. Current predict.py supports scalar or4-zone weights. Candidate predictions cached for later inexpensive calibration work. Third run smoke then full then three-way validation-only selection via run_zone_search.py. Nightwatch1fedfdaa7289f909083d30a3 verified running, supervisor2122856, GPU child2123090 on physicalGPU1. Remote /home/mnogales/Projects/HACKATON/zonal_campaign; output results/zone_search_gpu1; log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/1fedfdaa7289f909083d30a3.log. Initial ETA15m(~21:36Zurich). Latest log had advanced into full ridge candidates, proving smoke completed. A sandbox DNS lookup failure was retried with required escalated networking; remote state then retrieved normally. Next verify third terminal results and QA, retrieve/check chosen artifacts and assess diminishing returns. Goal remains active.
