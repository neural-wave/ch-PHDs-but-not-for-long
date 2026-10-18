---
title: GPU1 MAE search implemented, smoke verified, full campaign running
status: superseded
kind: experiment
recorded_at: '2026-10-16T18:51:31.839689+00:00'
sources:
- SEARCH_METHOD.md
- train_search.py
- reconstruction.py
- predict.py
- results/smoke_gpu1/metrics.json
- results/search_launch.json
experiment_id: null
supersedes: []
status_updated_at: '2026-10-16T19:22:49.677355+00:00'
---

User authorized minimizing error on Ando GPU1. Verified GPU1 idle and copied official data/code to /home/mnogales/Projects/HACKATON. Shared causal features, exact mean projection, ridge/ExtraTrees/absolute-error boosting,16 GPU neural configurations, validation-selected convex ensemble, all-data refit and offline inference implemented. Candidate list fixed before smoke output inspected; full configs unchanged after smoke. Split:first243 train/next60 validation/last60 holdout. Prior holdout observations may feed fixed earlier-history features for later dates without refitting, matching challenge inputs. Smoke holdout viewed; not a never-viewed test set. No secret targets or external data used.

Verified smoke:validation MAE3.9426197856 vs5.5721848687 baseline; holdout5.1090001618 vs6.6825855306 baseline. Max hourly deviation1.1369e-13; exported384-row CSV passed uniqueness/finite/four-quarters/hour/mean tolerance checks. Causal perturbation test passed. Smoke artifacts:remote results/smoke_verified_gpu1; local results/smoke_gpu1/metrics.json. Remote py_compile passed; local compile encountered macOS cache-write permissions. Earlier startup summary preserved verbatim in memory/records/20261016_startup_before_search.md.

Full launch verified2026-10-16 20:50Zurich:Nightwatch d663a92035ce2527ea276864, supervisor2112795, training2112796. CUDA_VISIBLE_DEVICES=1; nvidia-smi showed child on GPU1. Current installed Nightwatch CLI copied to execution host; existing private config, successful health, heartbeat_expected=true timeout180s, Telegram completion/crash alerts enabled. Worker log:/home/mnogales/.local/share/nightwatch-worker-bitsvsdims/d663a92035ce2527ea276864.log; outputs:/home/mnogales/Projects/HACKATON/results/search_gpu1/. Initial ETA1h(~21:50Zurich), not completion proof. Initial log completed ridge candidates. Next audit terminal scores/refit/inference QA, copy outputs locally, assess additional optimization. Goal remains active.
