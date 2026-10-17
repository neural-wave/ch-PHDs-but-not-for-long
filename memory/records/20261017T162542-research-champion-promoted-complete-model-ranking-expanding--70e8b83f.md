---
title: Research champion promoted, complete model ranking, expanding validation mandatory
status: current
kind: decision
recorded_at: '2026-10-17T16:25:42.140682+00:00'
sources:
- VALIDATION_PROTOCOL.md
- BEST_MODEL.md
- best_model.json
- MODEL_RANKING.md
- rank_models.py
- results/model_ranking_20261017/audit.json
- results/model_ranking_20261017/champion_per_fold.json
- results/model_ranking_20261017/sync_verification.json
- results/ensemble_normalization_20261017/backup_verification.json
- memory/records/20261017_startup_before_model_promotion.md
experiment_id: model_ranking_20261017
supersedes:
- memory/records/20261017T160737-expanding-minimization-finalized-ratio-0-708892-full-audit-a-87500786.md
---

On 17 October 2026 the user explicitly requested promotion of the winning model as best so far and a ranking of all tried models. They reaffirmed: always use the new validation method. VALIDATION_PROTOCOL.md now makes the eight expanding February–September tests, 3–10 fitting months, preceding-month days 1–15 selection and 16–end calibration, fresh fits, frozen test parameters, prior-only observed quarters, known full four-zone hourly curves, exact hourly means, eleven quantiles and equal-month MAE/mean-corrected-interpolation MAE the standing protocol. Other splits are diagnostics and cannot substitute for missing primary evidence. No new fitting or substantial prediction was performed for the ranking.

Canonical best_model.json and BEST_MODEL.md promote minimize_expanding_20261017/fixed_blend_ratio as the current research champion: mean monthly ratio 0.7079682867567635 (CSV aggregation differs from the owning report by 1.1e-16), 29.203171% less interpolation-relative MAE, pooled MAE 4.103184120698031 EUR/MWh, worst-month ratio 0.7542931676357179, calibrated 90% coverage 90.58718%, width 18.8354477. This is best observed developmental aggregate performance, not blind evaluation, mathematical optimality or an all-history production refit. The existing results/deployment_best.json inference bundle remains the earlier 363-day frozen pipeline. No Git remote or commits exist in this checkout; promotion is the research registry, with a verified copy on the research host.

Exact evaluation checkpoint configurations, weights and SHA256 hashes were exported on root@93.91.156.85:55340, /workspace/HACKATON using export_best_research_recipe.py without loading models. champion_per_fold.json covers eight folds and 51 checkpoint leaves. February blends 25% rich linear/75% MLP, March averages all six experts, April–September equally blends MLP/temporal CNN/analog 2D CNN. Each family can contain a validation-selected two-seed average. All leaves have positive weights summing to one per fold; training day counts and eleven quantiles checked. Champion archive remains local: results/minimize_expanding_20261017_completed.tar.gz, 2428656306 bytes, SHA256 6263d9223b7bcb7c9b4855ced55a74e4844ea1463d175866a5072d08236701d0. Owning audit passed 384 candidate rescores/281 selected checkpoint replays/92544 exported quarters.

rank_models.py consolidates ten audited eight-fold campaigns, recomputes every monthly ratio against identical official corrected-interpolation denominators, checks exact month/day sets, pooled MAE, source hashes and declared control aliases. It yields 142 reported rows, 110 primary ranked recipes after 32 explicit reused-control aliases, 1136 monthly rows, 121 historical recipe/scope rows, 5197 candidate source records and 43 campaign inventory rows. Candidate records can repeat a trial across campaign and per-trial sources; they are not independent model counts. Ten target-derived oracle diagnostics are unranked. September-only, 60-day holdout, fixed-September history diagnostics, superseded SSM pilots, pretrained FlowState, incomplete/smoke and deployment refits retain distinct scopes. No eight-fold score is invented for a raw HPO trial. Search budgets and inner objectives differ, so these are saved-recipe rankings, not controlled architecture superiority claims.

The strongest separately reported prior-data selection policy is ensemble_normalization_20261017/validation_selector, ratio 0.7083710434569991 and pooled MAE 4.106784906028097; its mixed-blend row ties it. The optimization campaign's own selector remains 0.7088921710561056. Cross-campaign choice of either policy is still developmental. Controlled six-expert extension backup_verification.json now exists with verified=true, 1555212494 bytes and SHA256 6acd7597146beaa22bcda64620f1af914ba42335f5b491641afe0bf368c35657; local archive size independently matches. This supersedes the startup's pending-backup wording, without inferring any process state from an ETA.

RUN_SOLUTION.md points to the required validation and new registry, while historical results/LEARNED_MODELS.md links to current evidence. Generated local Markdown links were checked. Eighteen promotion files were copied to /workspace/HACKATON and every remote SHA256 matched the local delivery manifest; sync_verification.json records the check. Ranking opened in Codex (queued by the app). Prior startup was preserved verbatim before shortening. Existing S4D results and paused classifier context remain in the startup. Future action is to use the prescribed validation for each new comparison; no training, production refit or Git push is pending under this promotion request.
