---
title: Hindsight hourly-win definition and proposed posthoc zone weight calibration
status: superseded
kind: proposal
recorded_at: '2026-10-17T14:38:48.738834+00:00'
sources:
- campaigns/monthly_quantiles_20261017/analyze_ensemble.py
- results/monthly_quantiles_20261017/ensemble_analysis/REPORT.md
- VALIDATION_PROTOCOL.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T14:50:52.764315+00:00'
---

2026-10-17: User asked what hindsight hour wins means and suggested calibrating zone ensemble weights as a posthoc procedure potentially applicable to other countries. Verified owning analysis: for each zone/hour, calculate each of six experts' q50 MAE across the four quarters against observed answers; argmin gets one win. The displayed share is the fraction of these zone-hours won, pooled by hour count for an all-month view. This is retrospective specialization evidence, not actual router choices or a forecast confidence probability. Win frequency need not order average MAE.

Proposed follow-up (not implemented, no run launched): fit nonnegative sum-to-one weights separately by zone using earlier labeled out-of-sample predictions/preceding selection days, preferably shrinking toward a global blend when data are limited. Use the same weights for all quarters within each hour and all quantiles, then calibrate final intervals on separate reserved days. Freeze before the next test month and retain equal-month MAE/interpolation-MAE as primary objective. Existing selection/calibration split remains days1-15 versus16-end of preceding month. A new country could reuse this procedure with compatible expert forecasts and labeled local calibration history; transfer of fitted weights or zero-shot country generalization has not been evaluated. Current completed results and fixed global-per-fold ensembles remain unchanged.

Preserved previous startup next-step wording: 'Next, if requested, fit an interpretable gate using earlier out-of-sample errors and recalibrate gated quantiles.' User's present interest favors a simpler posthoc per-zone convex blend before a feature router.
