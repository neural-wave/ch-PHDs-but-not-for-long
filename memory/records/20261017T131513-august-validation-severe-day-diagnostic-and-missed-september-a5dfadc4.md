---
title: August validation severe-day diagnostic and missed September worst day 20261017
status: current
kind: analysis
recorded_at: '2026-10-17T13:15:13.023279+00:00'
sources:
- inspect_validation_risk.py
- results/quantile_2d_cnn_20261017/validation_risk_diagnostic.json
- results/quantile_2d_cnn_20261017/risk_models.json
- results/quantile_2d_cnn_20261017/risk_predictions.npz
- results/quantile_2d_cnn_20261017/confidence_predictions.npz
- results/quantile_2d_cnn_20261017/trials/23/predictions.npz
experiment_id: quantile_2d_cnn_20261017
supersedes: []
---

User asked whether catastrophic errors are detected in validation. Recomputed daily errors directly from quantile-cohort trial23 point predictions and official targets, verified agreement with cached risk labels and exact dates. No forecasts, thresholds or selections changed.

Joint four-zone dayMAE>10 had zero events in all31August days. August1–15 selection meanMAE4.519094, worst August12 MAE6.444906. August16–31 calibration mean4.258319, worst August31 MAE6.025225. Primary rare-event detector is a constant fallback at probability0.009370389 for every day, so it does not discriminate or demonstrate catastrophe detection.

Separate continuous predicted-daily-MAE probe (inputs+uncertainty): August12 predicted5.446351 and ranked2nd of15 by predicted error; August31 predicted5.498387 and ranked1st of16. August ranking is diagnostic: first block selected the CNN and second calibrated the probe, so neither establishes independent generalization. Mean90% interval width correlated with actual dailyMAE at Pearson.855818(first15)/.787521(last16).

Important failure on September development data: worst day September5 had actualMAE7.243371, predicted dailyMAE4.807541, rank28th of30 (rank1 means highest predicted error). It was missed by the continuous warning signal. Across September width-vs-error Pearson.411466. Existing Spearman averages must not obscure this individual failure.

Daily averaging conceals point extremes: max absolute quarter error95.856188 in firstAugust block,62.298926 in lateAugust,111.508841 in September. Max zone-specific dayMAE8.004491,9.066488,15.378976 respectively. These are descriptive diagnostics, not new event definitions or detection evaluations. The current joint daily10 threshold cannot validate missed-spike detection.

Local reproducible script inspect_validation_risk.py writes validation_risk_diagnostic.json. Short artifact arithmetic ran locally in project-memory Python with NumPy; no local training or large research computation. Startup summary updated with the September miss. Next discuss a practically relevant operational event and more out-of-sample evidence before claiming reliable warnings.
