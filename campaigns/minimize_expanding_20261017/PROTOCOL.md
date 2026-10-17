# Minimize interpolation-relative error with expanding training history

Frozen 17 October 2026 before this campaign's outer-test evaluation. User request:
minimize the new validation metric, with an increasingly wide training set.
Follow VALIDATION_PROTOCOL.md exactly: eight February–September full-month tests;
training expands from October–December 2025 to October 2025–July 2026. Fresh
fitting per fold, preceding-month days 1–15 select, 16–end calibrate. No shortened
training windows, selection-month refit, test-time updates, external data,
pretrained components or October evaluation targets. Current hourly curves are
known; observed quarters are strictly earlier than the reconstruction day.

Primary objective: arithmetic mean of the eight monthly q50 MAE / same-month
mean-corrected linear-interpolation MAE ratios. Keep pooled original-currency
MAE, separate pooled ratio, worst month, monthly zone ratios and explicitly
named uncorrected-interpolation ratios. These are developmental backtests.

Run 384 new neural fits on the existing authorized Vast RTX 5090: six families
(rich linear, compact linear, compact additive, compact MLP, rich temporal CNN,
analog 2D CNN), ordinary and training-month-ratio objectives, learning rates
.001/.003 and seeds 17/43, eight folds. Same architecture/loss/budget as the
audited monthly_quantiles_20261017 comparison. The only fitted-model change is
removing fitted z-score normalization: retain original engineered physical
units and fixed [-15,15] clipping. Additive knots remain fitting-only. This
tests transfer of the successful normalization result to the stronger families.

Retain all eleven ordered quantiles. Loss MAE(q50) + mean eleven-quantile
pinball, batch 32, AdamW weight decay .01, gradient clip 5; maximum 300 epochs,
patience 35 (2D CNN 450/50). q50 conserves supplied hourly means. Ordinary and
ratio arms share configurations; ratio weights use fitting-month counts and
interpolation errors only. Every seed fits independently from scratch.

For each family/objective choose on selection q50 MAE among four single fits
and two equal-seed averages at the same learning rate. Reuse the original
audited z-score family winners and original blends as frozen controls: no
retraining is necessary for the preprocessing comparison. Verify their source,
data, selections and exact checkpoint replay. Do not fit on their outer errors.

Two fixed-scale blends and two combined fixed/z-score blends use a frozen
menu per objective: single expert; rich-linear pairs at .25/.5/.75; equal
top 2/3/4/6 experts ranked on the same selection dates. Combined menus also
include the original blend and the fixed-scale blend. Global weights are
constant for the entire test month and all zones/hours/quantile levels. A
pre-test selector compares all expert/blend recipes including the original
ordinary and ratio blends. No hourly router or zone weight fitting is added.

Freeze all model/blend selections before calibration or test diagnostics.
Separately calibrate every final quantile recipe on reserved days 16–end with
the established widen-only zone/quarter CQR and deterministic interval nesting.
Save raw eleven quantiles, calibrated 50/80/90/95% bounds and the selector's
keyed quarter export, width percentiles and width/error diagnostics.

Preflight first and last expanding folds, two epochs, all families/objectives;
verify future-target perturbation/truncation, fixed preprocessing, fitting-only
knots and weights, checkpoint replay, averaging identities, validation choices,
conservation, quantile ordering, CQR/calibration and output identity. Full
independent audit repeats every candidate score, selected checkpoint replay,
blend menu, interval and aggregate comparison. Nightwatch success requires that
audit, comparison report and a source/checkpoint/prediction recovery archive.
The original validation_selector (0.721285) and original ordinary blend
(0.720494) are historical references, not newly measured gains. The observed
best aggregate test recipe will remain distinct from the pre-test selector.

Execution: root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python.
Initial ETA 20 minutes including smoke, full audit and archive. Wait for the
existing S4D process to exit before taking the GPU; waiting is bounded at 30
minutes and an unexpected dependency state fails explicitly. All supervision
and actual completion/crash notifications run on that host through Nightwatch.
