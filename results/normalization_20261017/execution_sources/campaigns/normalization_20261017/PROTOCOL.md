# Matched normalization ablation — 17 October 2026

Follow VALIDATION_PROTOCOL.md: eight expanding February–September tests,
preceding-month days 1–15 select configurations/epochs, 16–end calibrate.
Fit preprocessing from scratch on fitting dates; freeze it throughout each test.
Official four-zone data only, strictly prior quarters, known current hourly curve.
September and all other reported months remain developmental backtests.

Three fixed model families: compact-summary CNN, S4D, S5. Identical compact
query features, seven-day/168-hour joint-zone histories, width32/state32,
learning rates .001/.003, seeds17/43, 300 epochs maximum, patience35, batch32,
AdamW, clip5. Four preprocessing arms, 48 fits/fold, 384 full fits total:

1. `fixed`: existing physical feature units (/100 for levels, /20 for
   differences/residuals), with no fitted feature standardization.
2. `standard`: existing fitting-set feature z-scores, scale floor .05.
3. `day_input`: per-query-day/zone price centering/scaling, then fitting-set
   z-scores. Fixed output residual scale20.
4. `day_reversible`: same inputs as day_input; undo daily scale on all output
   quantiles. This isolates output reversal from input normalization.

Daily mean mu and scale s=sqrt(population variance(hourly)+1) come ONLY from the
zone's supplied 24 hourly values. Normalize current level to (h-mu)/s, slopes,
cross-zone price differences and prior contrast summaries by s. Sequential
history interpolation residuals and absolute residuals use the same query-zone
s; prior hourly levels use (prior_h-mu)/s. Keep query mean/100 and std/100
side features in all arms to preserve absolute level and volatility information.
Calendar channels and availability masks are unchanged by daily normalization;
standardization retains existing mask restoration and zeroing of missing history.
Apply the same [-15,15] feature clip in every arm. Thus fixed means no fitted
z-score, not literal unscaled currency. This is a reversible-normalization
adaptation without learned affine parameters, not an exact RevIN replication.

Every model outputs eleven ordered quantiles, with a zero-mean constrained q50
residual over the mean-corrected interpolation anchor. Initialize gap biases
using fitting-only residuals in each arm's output units. Train every arm on the
SAME original-currency MAE + mean pinball objective (divided by the same constant
20); NEVER weight daily losses inversely by their daily scale. Select using
original-currency q50 MAE/interpolation MAE. Same parameter count per family.

Freeze a winner separately per family/arm before evaluating outer targets.
Also retain a preceding-block validation selector. Calibrate50/80/90/95% CQR
intervals separately for every recipe using the reserved block. Save quantiles,
intervals, width/error associations and keyed selector predictions. Primary
score: equal-month mean of eight q50 MAE/mean-corrected interpolation MAE ratios;
retain pooled MAE, distinct pooled ratio, per-zone/month scores, worst month,
pinball, coverage, width and explicitly named uncorrected-interpolation ratios.
Report paired arm differences for identical family/lr/seed, plus validation-
selected arm differences. No outer-test-driven new recipes or deployment update.

Required preflight: future-target perturbation and truncation causality, held-out
preprocessing invariance, independent affine feature checks, flat/negative curves,
first/last-fold two-epoch smoke, checkpoint replay, physical-loss equivalence,
quantile order, hourly conservation, CQR rank/nesting and metric recomputation.
All fitting and full data processing run on the existing authorized Vast GPU.
Nightwatch success requires the full independent audit and recovery archive.
