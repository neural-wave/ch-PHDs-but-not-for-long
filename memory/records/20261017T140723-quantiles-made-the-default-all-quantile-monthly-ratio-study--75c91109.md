---
title: Quantiles made the default; all-quantile monthly-ratio study launched and point
  campaign completed
status: superseded
kind: decision
recorded_at: '2026-10-17T14:07:23.799659+00:00'
sources:
- VALIDATION_PROTOCOL.md
- campaigns/monthly_quantiles_20261017/PROTOCOL.md
- campaigns/monthly_quantiles_20261017/trainq.py
- campaigns/monthly_quantiles_20261017/auditq.py
- results/monthly_quantiles_20261017/launch.json
- results/monthly_quantiles_20261017/smoke/audit.json
- results/monthly_ratio_20261017/REPORT.md
- results/monthly_ratio_20261017/audit.json
- results/monthly_ratio_20261017/backup_verification.json
experiment_id: monthly_quantiles_20261017
supersedes:
- memory/records/20261017T135818-monthly-ratio-minimization-campaign-launched-384-paired-neur-d159051c.md
- memory/records/20261017T131629-confidence-and-catastrophic-error-detection-set-aside-by-use-4a35c213.md
status_updated_at: '2026-10-17T14:24:41.369277+00:00'
---

User17October2026: "ah, lets use always quantiles. i wanna see later which predictions are confident and which arent". Set quantiles as the HACKATON default for subsequent predictive-model comparisons and persist in VALIDATION_PROTOCOL.md/startup. Reopens quantile uncertainty inspection; previously stopped catastrophic-error/risk-classifier training remains set aside. No new catastrophe/probability probe is included. Saved relative-width labels are not prediction-correctness probabilities, and empirical interval coverage does not establish individual error-warning reliability.

New campaign monthly_quantiles_20261017 trains all six families with positive pinball loss: rich/compact linear, compact additive/MLP, rich temporal CNN, globally hourly-conditioned analog2DCNN. Same two ordinary/month-ratio objectives,2LR,2seeds,8folds:384fits. Loss MAE(q50)+mean11-quantile pinball, with fitting-month baseline/day-count weights applied to both for ratio arms. Ordered levels .025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975, softplus cumulative gaps around mean-conserving q50. Nonmedian marginal quantiles need not conserve hourly means. Preceding-month days1–15 select epochs/models/blends by q50MAE ratio;16–end calibrate; following month tests. Strictly prior observations and full known four-zone24hourcurve. Source/feature snapshot from the prior campaign is frozen and hash-verified, with independent output directory.

Every recipe saves raw quantiles plus calibrated50/80/90/95%bounds. Widen-only CQR zone/quarter groups use corrected ceil((n+1)c) ranks and deterministic interval nesting. Save calibration corrections before test diagnostics. Validation-selected recipe exports92544keyedquarter rows in predictions.csv.gz, with11quantiles, calibrated90%bounds/width, midrank percentile relative to same-zone/quarter calibration widths, and narrow/middle/wide bins at .25/.75. Report actual bin MAE, same-bin interpolation MAE/ratio, coverage, width, pinball and width-error association. No thresholds chosen from outer outcomes. Convex quantile-function blends separately calibrated, not described as mixture-distribution quantiles.

Preflight first/last-fold smoke passed24candidate rescoring/24exactserializedreplays, meanerror5.684341886080802e-14,22272CSVquarterrows; audit independently verified source/feature/official identity, training-only stats/weights/knots, quantile order/nonzero widths, q50conservation, selection/blends, all calibration ranks/corrections/nesting/coverage, confidencepercentiles/bins/CSV.

Nightwatch5d8339ecf415745b4844ddb1 launched14:06:03UTC/16:06:03Zurich, initial15minute estimate16:21. root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python campaigns/monthly_quantiles_20261017/run.py. Worker8640; log/root/.local/share/nightwatch-hackaton/5d8339ecf415745b4844ddb1.log. Verifiedrunning with freshheartbeat, heartbeat_expectedtrue, Telegramenabledtrue and full training already in Marchfold. Completion requires fullquantileaudit and results/monthly_quantiles_20261017_completed.tar.gz. Full quantile results pending; no deployment replacement.

Prior point-only Nightwatchb6d61339b6b5b82131b36c0d completedexit0 at14:01:46UTC. Full8fold audit passed592candidate rescoring/96selected replays, replay0, maxmeanerror1.1368683772161603e-13. Best descriptive aggregate blend_ratio0.7204363333/pooledMAE4.1773896691; validation-only recipe selector0.7225560643/4.1886990662; blend_uniform0.7245915778/4.1980744028; compactlinear_ratio0.7303285799/4.2358377965. Lower than earlierlinear0.738303; no significance claim. Archived locally results/monthly_ratio_20261017_completed.tar.gz,543131682bytes,SHA256e2388507073b500ea54fc33136955e2a270418c7e1dcd468879ccd8e463042cf verified against remote receipt. All source/model artifacts preserved. Point-only results remain historical controls under the new quantile default.

Next verify the actual quantile job terminal state, final REPORT.md/audit.json/summary.csv and CSV, retrieve recovery archive and hash, then compare median ratios with calibration/width and confidence-bin errors.

## Verbatim previous startup

# HACKATON project memory

Objective: minimize average monthly MAE / interpolation_MAE for official four-zone quarter reconstruction. Preserve hourly means within 0.0001 EUR/MWh, strictly prior observed quarter history, official-only inputs and frozen inference parameters. No external/pretrained components or October evaluation prices in submissions. September remains developmental; zones anonymized; user prefers small interpretable CNNs.

Protocol: VALIDATION_PROTOCOL.md. Eight expanding February–September tests, three initial training months; next-month days1–15 selection, remaining days reserved for calibration. Train-only normalization/fitting. Primary score averages same-month model MAE / mean-corrected-linear-interpolation MAE equally. Fixed-September history sizes stay separate.

Active 2026-10-17: Nightwatch b6d61339b6b5b82131b36c0d, 384 neural fits across six families, ordinary versus monthly-ratio training; ridge/analog/blend controls. Relative loss weights use only fitting-month baseline errors and day counts. All learned models receive today's complete hourly curve. Source snapshot isolated in campaigns/monthly_ratio_20261017/sources/. Smoke audited: 44 candidates, 24 exact checkpoint replays, mean error≤1.14e-13. Full results pending. Host root@93.91.156.85:55340,/workspace/HACKATON; /venv/main/bin/python. Worker7626; log /root/.local/share/nightwatch-hackaton/b6d61339b6b5b82131b36c0d.log. Launched15:57Zurich; revised estimate16:04 after4/8folds. Heartbeat/Telegram enabled; success requires full audit/archive.

Prior benchmark: linear ratio0.738303/pooledMAE4.273508; linearquantile0.739333. September-only jointblend5.091209, analog90CNN5.177024, quantilecohort5.216651 lack matched monthly evidence. Offline363-day refit audited/no submission. Exploratory FlowState completed with no incremental gain; verified archive results/flowstate_20261017_completed.tar.gz.

Evidence: results/monthly_ratio_20261017/{launch.json,manifest.json,smoke/audit.json}; campaign PROTOCOL.md. Detail and verbatim prior startup: memory/records/20261017T135818-monthly-ratio-minimization-campaign-launched-384-paired-neur-d159051c.md. Next verify actual terminal state, full REPORT.md/audit.json, then retrieve recovery archive and hash; no deployment replacement.


Measured quantile progress at14:08UTC/16:08Zurich:3/8full folds complete, statusrunning/freshheartbeat. Revised Nightwatch estimate16:13Zurich (due_at1792246397.3883991), including audit/archive; no final quality claims yet. launch.json records both initial and revised estimates.
