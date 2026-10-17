---
title: 'Vast GPU posthoc zone quantile weight study completed: small shrinkage gain,
  all audits and backup verified'
status: current
kind: result
recorded_at: '2026-10-17T14:50:52.757335+00:00'
sources:
- campaigns/zone_weights_20261017/PROTOCOL.md
- campaigns/zone_weights_20261017/zone_blend.py
- campaigns/zone_weights_20261017/audit.py
- results/zone_weights_20261017/INTERPRETATION.md
- results/zone_weights_20261017/summary.csv
- results/zone_weights_20261017/zone_summary.csv
- results/zone_weights_20261017/weights.csv
- results/zone_weights_20261017/audit.json
- results/zone_weights_20261017/backup_verification.json
experiment_id: zone_weights_20261017
supersedes:
- memory/records/20261017T143848-hindsight-hourly-win-definition-and-proposed-posthoc-zone-we-f09becce.md
---

Verified 2026-10-17. User explicitly authorized trying posthoc zone ensemble weights on Vast.ai GPUs, while retaining quantiles. Also clarified that hindsight hourly wins compare frozen expert median forecasts with true held-out quarter prices during scoring: they count best-expert zone-hours, not ensemble choices or confidence; win frequency does not measure error magnitude.

Implemented campaigns/zone_weights_20261017/{zone_blend.py,audit.py,run.py,PROTOCOL.md}. Six frozen quantile experts per training objective, ordinary/uniform and monthly-ratio weighted, reused from the completed campaign. GPU full-batch objective/gradient optimization with simplex-constrained SLSQP fits median MAE; final smoothing0.0001EUR/MWh. Fourteen weight recipes plus one pre-test selector over eight unchanged chronological folds. Fitting uses preceding days1–15, calibration16–end, following whole month test. Shared weights, separate zone weights, zone/global fractions0/.25/.5/.75/1, common-selected fraction, and zone-selected fractions compared. Fractions use fitting1–10/scoring11–15 then refit1–15; objective/policy selector uses that tuning score. Experts were previously checkpoint-selected using all15 days, so the inner tuning score is not independently nested. All test labels excluded from weight fitting/selection. All quarters and eleven quantiles share a zone vector; final CQR calibration is separate per blend.

Results (equal-month MAE/interpolation-MAE, pooledMAE EUR/MWh, empirical calibrated90%coverage, meanwidth):
- Existing ordinary blend .7204941118854624 / 4.1764877276085945 / .9230095954356846 / 20.751523382634232.
- Continuously fitted shared weights ordinary .7193106862102625 / 4.167925212540552 / .9169692254495159 / 20.087128305826806.
- Fully separate zone weights ordinary .7204853102964297 / 4.1753605795117155 / .9132844917012448 / 19.92177450243517.
- Best observed grid recipe, ordinary 75% fitted shared +25% fitted zone weights .7189066090196203 / 4.165800762338455 / .9166018326417705 / 20.018773665283927. Relative ratio gain .2203353% versus original blend, only .0561756% beyond new shared fit; pooledMAE drop .01068697EUR/MWh. Improves4/8 monthly ratios. Best aggregate recipe is identified posthoc; no statistically established gain claimed.
- Pre-test objective/fraction-policy selector .7199524302721731 / 4.1734458684954046 / .9213563278008299 / 20.8382535147294. Prior corresponding selector .7212852815039343 / 4.181171866093176.

Unrestricted zone weights improve ZONE_2 conditional ratio .836246 -> .826430, while worsening ZONE_0/.1. Shrinkage tempers those losses. New selector narrow/middle/wide MAE2.3570/3.4802/6.4788, showing relative widths still track absolute difficulty. Country transfer not run; reusable procedure requires compatible frozen experts plus local labeled fitting/calibration history, not unchanged country weights. No hour-dependent router or production replacement.

All120 blend/fold replays passed, 92544 exported selector quarters, max quantile reconstruction discrepancy2.27e-13, max hourly-mean error1.14e-13EUR/MWh. Ten independent first-fold exact LP checks verify GPU fitting MAE within9.78e-8EUR/MWh. Official-data/source identity, chronology, simplex constraints, quantile ordering, separate interval correction/nesting/coverage, confidence labels, monthly scores, matched-control parity and CSV identity audited.

Execution host root@93.91.156.85:55340, /workspace/HACKATON, RTX5090 UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9, /venv/main/bin/python. Nightwatch3b0716f2e5242db0fb5b1e04 completed exit0 at14:48:22UTC/16:48:22Zurich; runtime approximately55seconds (estimate8minutes). Heartbeats and actual success/crash notifications enabled; remote current CLI matched installed SHA414d11592b7a5edd46e90fa913dd79fad4dbc6826975327aca4c17b8702343f5. Log /root/.local/share/nightwatch-hackaton/3b0716f2e5242db0fb5b1e04.log. No active job remains from this experiment.

Full archive results/zone_weights_20261017_completed.tar.gz copied locally and independently SHA256/size verified:247761378bytes, SHA35944c95e24e176c8423aefd6d02dda5ff326b1c0d86ee34ce58fc92e0172a0f. Archive includes all new per-recipe quantiles/intervals, decisions, source hashes and code; source frozen expert checkpoints remain in the earlier verified monthly_quantiles archive. Detailed interpretation in results/zone_weights_20261017/INTERPRETATION.md; summary.csv, monthly_scores.csv, zone_summary.csv, weights.csv, confidence_bins.csv, audit.json, optimization_audit.json, predictions.csv.gz and backup_verification.json hold evidence. No submission/deployment change.

Next step only if requested: evaluate a longer-history weight-fitting/calibration scheme or a small interpretable hour-dependent gate under a new frozen chronological protocol. Current evidence favors modest shrinkage rather than free zone weights; country-transfer performance is unmeasured.

## Preserved previous startup summary

# HACKATON project memory

Objective: minimize equal-month MAE / mean-corrected-interpolation MAE. User requires quantiles by default: eleven levels .025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975, calibrated50/80/90/95%intervals. Conserved q50 supplies point MAE. Confidence labels describe relative width, not correctness probabilities. Catastrophe/risk classifiers remain set aside.

Constraints: official-only four-zone data, no external/pretrained components or October evaluation prices; strictly prior observed quarters, full known current hourly curve, frozen inference parameters, hourly means within0.0001EUR/MWh. September developmental; zones anonymized; interpretable small models preferred. Eight February–September expanding tests; preceding-month days1–15 selection,16–end calibration; training-only statistics/weights. VALIDATION_PROTOCOL.md.

Verified2026-10-17: all384quantile fits completed/audited. Best blend ordinary-training ratio0.720494, pooledMAE4.176488,90%coverage92.30%,width20.75EUR. Validation selector0.721285. Bestsingle compactlinear_ratio0.730386. Export92544quarters. Narrow/middle/wide selector MAE2.33/3.60/6.50 also tracks baseline difficulty. Quantile Nightwatch5d8339ecf415745b4844ddb1 completedexit0 at16:12Zurich; point b6d61339b6b5b82131b36c0d completed. Host root@93.91.156.85:55340,/workspace/HACKATON; logs/root/.local/share/nightwatch-hackaton/{ID}.log. Quantile1.313GBarchive and point543MBarchive locally SHA256verified.

Ensembles use fixed validation-selected weights per test month for every input/quantile, then separate calibration. No learned router. Analog2DCNN winsZONE_2 in7/8months under both objectives; curve rankings less stable. Ordinary blend beats all6experts on about5.4%hours. Hindsight whole-hour oracle0.556926 is NOT deployable; no gate trained. Interpretive view and conditional audits: results/monthly_quantiles_20261017/ensemble_analysis/.

Evidence: results/monthly_quantiles_20261017/{REPORT.md,audit.json,backup_verification.json,predictions.csv.gz}; completed archives in results/. Detail/verbatim prior startup: memory/records/20261017T142334-all-quantile-campaign-completed-and-backed-up-fixed-ensemble-5a1e84d3.md. Proposed next: posthoc zone-specific convex weights from earlier validation forecasts, separate quantile calibration; country reuse needs local labeled history. No zone-weight fit yet. FlowState remains exploratory/no gain; offline363-day refit audited/no submission. No deployment replacement.
