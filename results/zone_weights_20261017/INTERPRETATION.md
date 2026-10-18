# What the zone-weight experiment shows

Completed on the authorized Vast.ai RTX 5090 on 17 October 2026. Each fold
reuses six frozen experts with all eleven quantiles. Posthoc weight fitting
uses preceding-month selection predictions and labels only. Calibration uses
separate days16–end; evaluation is the following full month. No trained expert,
production model, or submission was replaced.

| Method, ordinary-training experts | Mean monthly MAE/interpolation MAE | Pooled MAE, EUR/MWh | Calibrated 90% coverage | Mean 90% width |
|---|---:|---:|---:|---:|
| Original validation-menu blend | 0.720494 | 4.176488 | 92.30% | 20.7515 |
| Continuously fitted shared weights | 0.719311 | 4.167925 | 91.70% | 20.0871 |
| Fully separate zone weights | 0.720485 | 4.175361 | 91.33% | 19.9218 |
| 75% fitted shared + 25% fitted zone weights | 0.718907 | 4.165801 | 91.66% | 20.0188 |

The best observed grid recipe reduces the ratio by 0.22% relative to the
original blend. Most of that gain comes from continuously fitting the shared
weights (0.16%); adding a small zone adjustment gives another 0.056% relative
to that shared fit. Its pooled MAE reduction is about 0.011 EUR/MWh. This is a
small developmental improvement, not evidence of a statistically established
gain. It improves four of the eight monthly ratios and worsens the other four.
The best grid recipe is identified after reporting all test outcomes; it must
not be presented as a recipe selected before seeing these outcomes.

The separate policy that selects training objective and zone shrinkage using
only pre-test tuning dates obtains ratio0.719952, pooledMAE4.173446,
90%coverage92.14%, and width20.8383. Its corresponding original validation
selector was ratio0.721285 and pooledMAE4.181172. Expert checkpoints already
used all15 selection days, so the internal 1–10/11–15 split is a tuning device,
not independent nested validation. The outer test chronology is preserved.

Unrestricted zone fitting helps ZONE_2 most: its ordinary-training conditional
monthly ratio moves from0.836246 (original blend) to0.826430. It worsens
ZONE_0 and ZONE_1, explaining why unrestricted zone weights barely improve the
overall score. The 25% zone adjustment tempers those losses and retains some
ZONE_2/ZONE_3 gains. See zone_summary.csv and weights.csv for exact values;
weights change by test fold and never use that fold's test labels.

For the pre-test selector, narrow/middle/wide calibrated-width groups have
pooled MAE2.36/3.48/6.48 EUR/MWh. Relative interval width remains informative
about absolute error; these labels are not probabilities of correctness.
All saved blends preserve median hourly means and ordered quantiles, with
separate calibrated50/80/90/95% intervals. Coverage is empirical under time
dependence, not a guaranteed next-month coverage rate.

The procedure is posthoc: compatible frozen expert predictions and local
labeled historical data suffice to fit weights and calibrate intervals.
Country transfer has not been tested, and these fitted zone weights should
not be assumed to transfer unchanged to another country.

Audit: all120 recipe/fold replays pass; 92,544 selector quarters exported;
maximum hourly-mean error1.14e-13 EUR/MWh. Independent exact first-fold linear
programs differ from the GPU-fitted MAE by at most9.78e-8 EUR/MWh.

Evidence: REPORT.md, summary.csv, monthly_scores.csv, zone_summary.csv,
weights.csv, confidence_bins.csv, optimization_audit.json, audit.json,
folds/*/selection.json and folds/*/selected/*/predictions.npz.
Nightwatch3b0716f2e5242db0fb5b1e04 completed with exit0 at14:48:22 UTC.
Host root@93.91.156.85:55340; project/workdir /workspace/HACKATON;
log /root/.local/share/nightwatch-hackaton/3b0716f2e5242db0fb5b1e04.log.
