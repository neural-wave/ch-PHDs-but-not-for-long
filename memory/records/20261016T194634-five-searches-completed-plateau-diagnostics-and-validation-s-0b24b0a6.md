---
title: Five searches completed, plateau diagnostics and validation-size discussion
status: superseded
kind: experiment
recorded_at: '2026-10-16T19:46:34.776376+00:00'
sources:
- results/LEARNED_MODELS.md
- results/zone_search_gpu1/metrics.json
- results/catboost_search_gpu1/metrics.json
- results/projection_search_gpu1/metrics.json
- results/best_search.json
- results/artifact_sha256.txt
- results/paired_diagnostics.json
- RUN_SOLUTION.md
- predict_best.py
- select_best.py
- train_catboost.py
- catboost_models.py
experiment_id: null
supersedes:
- memory/records/20261016T193112-zone-weight-preliminary-gain-isolated-gpu-mae-boosting-famil-f2f3761c.md
status_updated_at: '2026-10-16T20:00:43.006280+00:00'
---

Previous goal turn made progress (new family prepared/queued). Live evidence this turn: zone tracker1fedfdaa7289f909083d30a3 and CatBoost0d5642c9768c674cf8f2047a completed with exit0 and final eligibility QA; fifth constrained-correction refinement76220b995fab9750c2f78305 also completed exit0. No active training remains. All substantive fitting ran remotely on physical AndoGPU1, including verified fifth child2135331 using12224MiB. Remote root /home/mnogales/Projects/HACKATON; durable logs /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/<runid>.log. Tracking heartbeats and actual completion/failure alerts were enabled.

Verified validation/holdout MAE: zone3.6647421172/4.7813136190; CatBoost expansion3.6637283687/4.7798611490; correction refinement3.6644300569/4.7808313261. Final manifest picks CatBoost by validation only across five complete pipelines. Baseline holdout6.6825855306, observed improvement28.47%. Fifth did not beat existing winner. Fifth trained up to6promising/selected prior configurations at fixed selected tree counts; adds per-zone nonnegative correction coefficients summing1, fitted first30validation days, accepted only if improved latter30, then fitted all60. Uniform alternatives retained. Numerical constraint/uniform-equivalence/holdout-target-poisoning tests passed. All exported full-model CSV eligibility checks passed.

Retrieved first four final models plus metrics/heldout arrays; fifth losing model remains preserved remotely, metrics/arrays retrieved. Remote/local model SHA256 match: zone5a159260592d3273d576e463e96306ed12cd970691c38ca194f134fa632ca746; CatBoost6a6b8b50ab3bf84cd506ba4a0dc2a92af6b83e61fae3512b4b4c834537585a20. Earlier hashes remain in results/artifact_sha256.txt. Added snapshots campaigns/zonal, campaigns/catboost, campaigns/projection. New frozen-manifest entrypoint predict_best.py cold execution on Ando matched original winner's exported CSV byte-for-byte. select_best.py selects completed QA-passing reports by validation only and requires chosen artifact present; missing losing artifact is allowed. RUN_SOLUTION.md gives full training sequence and offline command/dependencies. No new external prices/weights used; CatBoost isolated in project vendor directory.

Remote 10000-replicate paired7-day block bootstrap over already-viewed60-day holdout: baseline-to-CatBoost daily MAE gain1.902724 with descriptive95% interval[1.742889,2.116306]. Incremental zone-to-CatBoost gain0.001452, interval[-0.000853,0.003992], suggesting practical plateau, not established secret-test improvement. Holdout diagnostics are not used for manifest selection.

User asked validation size and questioned60 days as large. Verified original split: training2025-10-01..2026-06-02(243days); validation2026-06-03..2026-08-01(60days), four zones,23040quarters/5760hours; holdout2026-08-02..2026-09-30(60days). Explained60validationdays=16.5%, another60holdout leaves67%candidate fitting; conservative starting choice against volatile-price-day selection noise, not challenge requirement or optimized split. Suggested rolling28-day windows as next comparison; no changed split was applied or explicitly requested yet. Final all-data artifacts already fit all363days, so split affects selection experiments only. Goal remains active to address validation concern and consider rolling comparison; no completion claim.
