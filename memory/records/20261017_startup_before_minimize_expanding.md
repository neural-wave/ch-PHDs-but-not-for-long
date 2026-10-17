# Verbatim startup before expanding optimization

# HACKATON project memory

Objective: minimize equal-month q50 MAE/mean-corrected-interpolation MAE. Normalization study completed. Default eleven quantiles .025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975 and calibrated50/80/90/95%intervals. Width labels describe relative uncertainty; catastrophe classifiers remain set aside.

Constraints: official four-zone data; no external/pretrained components or October evaluation prices. Strictly prior quarters, known current hourly curves, frozen parameters, hourly means within0.0001EUR/MWh. Eight expanding February–September developmental tests; preceding-month days1–15 selection,16–end calibration; fitting-only preprocessing. VALIDATION_PROTOCOL.md.

Verified2026-10-17: normalization384fits/eightfolds completed/audited,replay0. Fixed engineered scaling beats fitting z-scores: compactCNN.769238vs.797334,S4D.763497vs.782518,S5.762926vs.789864;2.4–3.5%relative gain,7/8months each,88/96matched pairs. Daily input/reversible scaling worse;reversible90%widths+11–24%. BestS5fixed pooledMAE4.422713,90%coverage89.37%. Fixed retains/100levels,/20differences,clip15. No universal normalization claim;stronger ensemble still better.

Scientific Nightwatch73994759a86ac27184894e2a completedexit0,17:11Zurich. Host root@93.91.156.85:55340,/workspace/HACKATON; log/root/.local/share/nightwatch-hackaton/{ID}.log. Evidence results/normalization_20261017/{REPORT.md,audit.json,insights.json};exact24training sources hash-verified in execution_sources/. Local1.113GBarchive SHA256verified: backup_verification.json. Backup Nightwatch8fdc7710e2b4dcb2e26b4158 completedexit0 at17:18Zurich;log/Users/miguelnogales/Documents/Codex/.nightwatch/workers/{ID}.log. Stronger-family fixed-scale follow-up proposed,not launched.

Prior quantile/zone study audited/backed-up: original blend.720494, best posthoc shrinkage.718907, pre-test selector.719952; inner tuning not independently nested. results/zone_weights_20261017/INTERPRETATION.md. No measured country transfer/hour router. FlowState no gain; offline363day refit audited/no submission. No deployment replacement.

Corrected SSM160fits completed/audited/backed-up: TCN.778942,S4D.785554,S5.788695. Nightwatchf1471cb72fbc1d8ade56973a; results/ssm_monthly_20261017/{REPORT.md,audit.json,backup_verification.json}. Old56fit pilot exploratory.

Detail: memory/records/20261017T151810-normalization-completed-fixed-physical-scaling-wins-across-c-dafc5839.md. Prior startup preserved verbatim in memory/records/20261017_startup_before_normalization_results.md.

SSM: nonlinear160fit comparison audited/backedup; ratios S4D.785554/S5.788695/TCN.778942(results/ssm_monthly_20261017). One-layer affine S4D frequency study running: Nightwatch d85b1d1b1b58a45af40c3a57,sameVast;log/root/.local/share/nightwatch-hackaton/{ID}.log.128fits,eight-fold quantiles,learned/frozen frequencies. Exactlinearity and repaired first/lastfold smoke passed;fullauditpending. Protocol campaigns/linear_s4d_20261017/PROTOCOL.md;receipt results/linear_s4d_20261017/launch.json. Next retrieve/audit/back up. Prior startup: memory/records/20261017_startup_before_linear_s4d.md.
