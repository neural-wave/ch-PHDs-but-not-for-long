# HACKATON project memory

Objective: minimize equal-month q50 MAE/mean-corrected-interpolation MAE; currently test normalization. Default eleven quantiles .025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975 and calibrated50/80/90/95%intervals. Width labels describe relative uncertainty; catastrophe classifiers remain set aside.

Constraints: official four-zone data; no external/pretrained components or October evaluation prices. Strictly prior quarters, known current hourly curves, frozen parameters, hourly means within0.0001EUR/MWh. Eight expanding February–September developmental tests; preceding-month days1–15 selection,16–end calibration; fitting-only preprocessing. VALIDATION_PROTOCOL.md.

Verified2026-10-17: fitting-set standardization already used; old September-only hybrid RevIN benefit small/inconsistent (REVIN_PROTOCOL.md). New384fit comparison: fixed engineered units, fitting z-scores, daily inputs, daily reversible input/output scaling; compact CNN/S4D/S5, paired rates/seeds. Daily statistics input-only; identical physical-currency loss. First/last-fold24checkpoint independent smoke audit passed, replay difference0.

Active Nightwatch73994759a86ac27184894e2a; started17:03Zurich, ETA17:18. Host root@93.91.156.85:55340,/workspace/HACKATON; log/root/.local/share/nightwatch-hackaton/{ID}.log. Running/heartbeats/training verified; full audit/archive pending. Evidence campaigns/normalization_20261017/PROTOCOL.md; results/normalization_20261017/{launch.json,smoke/audit.json}. Next retrieve paired results and hash-verify archive after completion.

Prior quantile/zone study audited/backed-up: original blend.720494, best posthoc shrinkage.718907, pre-test selector.719952; inner tuning not independently nested. results/zone_weights_20261017/INTERPRETATION.md. No measured country transfer/hour router. FlowState no gain; offline363day refit audited/no submission. No deployment replacement.

Corrected SSM160fits completed/audited/backed-up: TCN.778942,S4D.785554,S5.788695. Nightwatchf1471cb72fbc1d8ade56973a; results/ssm_monthly_20261017/{REPORT.md,audit.json,backup_verification.json}. Old56fit pilot exploratory.

Detail: memory/records/20261017T150451-normalization-audit-and-matched-four-arm-eight-month-quantil-4cf6c4ca.md. Prior startup preserved verbatim in memory/records/20261017_startup_before_normalization{,_after_ssm_completion}.md.
