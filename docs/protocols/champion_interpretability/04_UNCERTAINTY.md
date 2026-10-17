# 4. Disagreement, uncertainty and failure atlas

Question: does expert disagreement add information beyond calibrated interval width?

Extend existing uncertainty and cancellation tables using weighted expert-median dispersion, raw quantile spread and calibrated 50/80/90/95% width. Fix bin edges on each fold's calibration data, then apply unchanged to its test month. Report MAE, pinball loss, coverage, width and counts in width-by-disagreement bins, by month and zone. Distinguish pre-calibration spread from calibration expansion.

Inspect disagreements with narrow final intervals and shared expert failures with wide intervals. Compare equal-month within-bin model/interpolation ratios on identical targets; undefined or empty groups remain explicit. Width is not a probability of correctness, and dependence among experts prevents treating dispersion as a posterior variance.

Create an atlas with predeclared typical cases, plus separately labelled hindsight biggest misses/cancellations. No risk classifier, routing policy or automatic rejection rule. If constructing a new interval method, reserve chronological fitting/calibration and evaluate it as a separate predictive comparison.

Acceptance: calibration-only bins, all interval levels retained, sparse groups disclosed, no re-use of test errors to set thresholds.

See [shared protocol](PROTOCOL.md). Status: proposed, not executed.
