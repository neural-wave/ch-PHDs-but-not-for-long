# Normalization of the stronger six-expert ensemble — 17 October 2026

The previous normalization table covered compact-summary CNN/S4D/S5, not the
stronger six-expert ensemble. Its conclusions cannot be applied to that ensemble.
The original ordinary-training quantile blend scored .7204941118854624; the
later .718906609 zone-shrinkage recipe is a separately reported posthoc study.

Compare fixed engineered scaling with fitting-only z-scores for ALL six original
families: linear_rich, linear_compact, additive_compact, mlp_compact, cnn_rich,
cnn2d_analog. Reuse exactly the original ordinary-training source snapshot, raw
features, architecture, quantile parameterization, rates .001/.003, seeds17/43,
MAE+pinball loss, output scale20, batch32, optimizer, epoch/patience limits.
Fixed means skip fitted z-scores, retaining original /100 and /20 physical units
and the common [-15,15] clip. Append identical compact zone one-hot indicators.
Fit additive knots on fitting data in the relevant feature coordinates.
This is an input-preprocessing ablation, not a daily RevIN comparison.

Eight expanding February–September developmental tests follow
VALIDATION_PROTOCOL.md: preceding-month days1–15 select,16–end calibrate;
no refit on those dates; frozen test parameters; strictly prior quarter history.
The original raw features already passed future perturbation/truncation audits.
Eleven ordered quantiles, hourly-conserved q50, CQR50/80/90/95 and per-quarter
exports remain required. No deployment update or new data is authorized here.

192 NEW fixed-scaling fits;192 exactly matched ordinary-training standardized
controls reused from the completed/audited original campaign. Reuse is documented
with source checkpoint/prediction hashes and exact prediction identity. First/last
fold smoke instead trains both arms for two epochs,24fits, to check new plumbing.
No existing source, original checkpoint, normalization study or active job is
modified. Full control replay must recover original blend ratio .7204941118854624.

Select family winners separately in each arm using only preceding first15days.
For each arm, use the SAME original blend menu: individual experts, rich-linear
pairs with weights .25/.5/.75, equal-weight validation top3. Also preregister a
mixed blend: select the normalization arm separately per family by first15day
MAE, then apply that same six-expert menu. Retain all pure controls and a
validation-only selector across declared recipes. All blend choices freeze
before calibration/test scoring and each blend calibrates independently.

Report mean monthly q50MAE/mean-corrected-interpolation MAE, each month/zone,
worst ratio, pooledMAE and distinct pooled ratio, pinball, interval coverage/width,
width/error diagnostics, and paired same-family/rate/seed differences. Retain
alternate uncorrected-interpolation ratios with explicit names. Never compare
the earlier compact-summary CNN to the original rich-feature temporal CNN as
though they were the same model. Quantile convex averages are quantile-function
blends, not quantiles of mixture distributions.

Run only on the existing authorized Vast GPU; wait durably for the affine S4D
job d85b1d1b1b58a45af40c3a57 to reach an actual terminal state before GPU work.
Execution-host Nightwatch tracks queue/training/audit/archive success or crash.
Preflight and full independent audits verify train-only preprocessing/knots,
control identity, all checkpoint replay, selection/blend menu, chronology,
conservation/order, calibration ranks/nesting, score/CSV identity. Success
requires full audit and recovery archive, never an expired ETA.
