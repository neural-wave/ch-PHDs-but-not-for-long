---
title: User-requested August validation September test implemented and launched
status: superseded
kind: experiment
recorded_at: '2026-10-16T20:00:43.000848+00:00'
sources:
- MONTHLY_PROTOCOL.md
- RUN_SOLUTION.md
- train_search.py
- train_catboost.py
- run_monthly.py
- predict_best.py
- results/monthly_launch.json
- train_rolling.py
- rolling_plan.json
experiment_id: null
supersedes:
- memory/records/20261016T194634-five-searches-completed-plateau-diagnostics-and-validation-s-0b24b0a6.md
status_updated_at: '2026-10-16T20:12:17.099272+00:00'
---

Previous goal turn made progress by completing/auditing five searches and discussing split size. This turn initially implemented a28-day rolling comparison and launched Nightwatch7414cb7764019baafdd79a4f on AndoGPU1. Its reduced smoke subset had zero prior-anchor columns for some zones, producing NaNs and failing mean-compliance assertion. Tracker authoritatively reports crashed exit1; wrapper absent, full rolling search never started. Fixed normalization by giving empty columns uniform weights before normalization. Did not restart because user then explicitly requested one month validation and another test. Explained failure candidly; retained crash/log for provenance.

Latest explicit user steering: train throughJuly, validateAugust, testSeptember. Verified counts from official CSV302/31/30days;11904August and11520September quarter targets. Implemented --calendar-split and --defer-refit in train_search.py; parent-driven split counts and deferred-refit artifacts in train_catboost.py. Training gradients/normalization fit302days, early stopping and mixtures select onlyAugust; test scores never select the pipeline. Saved fitted candidates rather than retraining onSeptember. Final report flags model_fit_days302 and final_refit_deferred=true; wrapper asserts them and exactsplit counts. New calendar outputs use test_predictions.npz; historical holdout metric field is retained as backward-compatible alias to September's test score. Earlier-date history withinAugust/September can feed fixed features on later dates, as official jury-history rules permit; no inference-time fitting. September was previously viewed during development, so not a fresh blind test.

User also asked whether fourcountries/nodes enter simultaneously. Verified reconstruction.py includes h.ravel() (4zones×24hourly prices), cross-zone neighbouring curves/differences and zone identifiers in every shared row. Models predict one zone/hour's fourquarter residuals at a time. Current neural nets are MLPs; regressions/trees share parameters acrosszones. No graph neural network claimed.

Monthly pipeline staged /home/mnogales/Projects/HACKATON/monthly_campaign. run_monthly.py runs base smoke, full31-candidate family, CatBoost smoke and11-config GPU boosting comparison, preserving302-day fit throughout, then writes results/monthly_best.json byAugust validation MAE only. predict_best.py supports that manifest and now defaults to monthly_best.json. Historical manifests/code remain accessible; no incomparable old/newvalidation-score selection. Remote py_compile passed; GPU1 idle and Nightwatchhealthokay beforelaunch. Current Nightwatch CLI refreshed frominstalledsource.

Launched Nightwatch33ca46c2ab0f4ebb14e99ea2 (~21:57Zurich), ETA20min(~22:18), supervisor2147568, verifiedGPUchild2147754. Heartbeat expected180s and Telegramcompletion/crash alerts enabled. Log:/home/mnogales/.local/share/nightwatch-worker-bitsvsdims/33ca46c2ab0f4ebb14e99ea2.log. Latest tracker running with fresh heartbeat; log reached full ridge1..10000 candidates, proving initial smoke/eligibility/split/302-day assertions passed. Outputs results/monthly_base_gpu1 and monthly_cat_gpu1 under remoteproject root. MONTHLY_PROTOCOL.md and updated RUN_SOLUTION.md distinguish current protocol from historical searches. Next audit terminal metrics/QA, ensure test exclusion, retrieve matched-model artifacts. Goal remains active.
