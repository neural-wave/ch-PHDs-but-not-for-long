---
title: Zone-weight preliminary gain; isolated GPU MAE boosting family queued
status: superseded
kind: experiment
recorded_at: '2026-10-16T19:31:12.015246+00:00'
sources:
- train_catboost.py
- run_catboost_after.py
- catboost_models.py
- predict.py
- requirements-catboost.txt
- results/catboost_launch.json
- SEARCH_METHOD.md
- project_memory.md
experiment_id: null
supersedes:
- memory/records/20261016T192249-two-full-searches-audited-and-retrieved-zone-specific-refine-3fd368d6.md
status_updated_at: '2026-10-16T19:46:34.784208+00:00'
---

Previous goal turn was progress: two completed runs audited/retrieved and third search launched. Current live checks confirmed third wrapper2122856 still running/refitting. Its metrics currently show validation3.6647421172 and holdout4.7813136190, mean error2.27e-13, but offline QA is not yet present: do not report complete. Best fully verified/retrieved model remains rich_search_gpu1 (validation3.6760608911/holdout4.7872780226) until third finalization.

Prepared a distinct GPU absolute-error boosting family. Installed CatBoost1.2.8 and graphviz0.21 in /home/mnogales/Projects/HACKATON/catboost_vendor with --no-deps, leaving shared conda environment unchanged; CatBoost import/version passed. Official CatBoost documentation confirms MAE GPU support; only library/code documentation consulted, no external challenge prices or weights. Added train_catboost.py, catboost_models.py, recursive parent support in predict.py, requirements-catboost.txt and run_catboost_after.py. Staged sources under /home/mnogales/Projects/HACKATON/catboost_campaign; remote py_compile passed. Original live zonal_campaign imports unchanged.

Plan implemented: wait on tracked third wrapper's PID AND run-id command identity, require final offline QA, then3-tree pooled smoke and full9pooled+2quarter-specific CatBoost configurations. Rich features and original chronological243/60/60 split unchanged. New tree counts selected by validation early stopping, all selected models refit on363official dates. Parent candidate reconstructs third ensemble's heldout/validation predictions from trusted selection cache; asserts source hash, date/target equality and exact reported validation MAE. New models plus parent/interpolation receive global then zone-shrunk convex weights, validation only. Chosen final parent artifact is frozen all-data refit; nested inference must pass exported384-row CSV eligibility checks. Finally best_search.json compares four complete pipelines by validation only. No secret-test claims; holdout already inspected in development.

Nightwatch0d5642c9768c674cf8f2047a launched and verified running on Ando, supervisor2125211; log repeatedly waits for verified-live third wrapper. ETA45min from~21:27Zurich (~22:12), heartbeat_expected=true timeout180s; success/failure alerts arranged. Log:/home/mnogales/.local/share/nightwatch-worker-bitsvsdims/0d5642c9768c674cf8f2047a.log; outputs:/home/mnogales/Projects/HACKATON/results/catboost_search_gpu1/. GPU1 is selected via CUDA_VISIBLE_DEVICES=1 and CatBoost devices0 within that visibility. Queue avoids concurrent experiments on same GPU. A read-only complex SSH result query hit sandbox DNS failure and succeeded when retried with required network escalation; simple SSH continued normally. Next audit third QA/terminal tracker and fourth smoke/launch, retrieve final artifacts only after completion, compare performance/diminishing returns. Goal remains active.
