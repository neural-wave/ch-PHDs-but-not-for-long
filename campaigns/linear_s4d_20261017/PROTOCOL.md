# One-layer interpretable S4D frequency study — 17 October 2026

User asked for a very simple S4 layer to make predictions interpretable and learn
frequencies. This uses S4D (diagonal S4), not full DPLR S4. Existing nonlinear
SSM comparison is preserved. Run on existing Vast RTX5090
root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python.

## Exact linear point path

768 quarter-hour tokens: seven prior96-quarter days plus current96 known hourly
values repeated four times. Ten channels: four hourly prices, four signed past
interpolation errors, known-hourly flag, observed-quarter flag. Current quarter
errors are exactly zero/unavailable. Missing calendar days retain zero masked
blocks. No absolute-error features, nonlinear feature engineering, clipping,
activation, LayerNorm, gating, MLP or learned encoder projection.

One S4D instance with10 explicitly named input channels and8/16 real-equivalent
states per channel (4/8 complex damped oscillations). Share it in forward/reverse
directions; all inputs in this complete window are permitted known values.
Concatenate the two linear outputs, linearly read out four quarter-price
corrections, then subtract their mean within each hour and add the conserved
interpolation baseline. This point mapping is affine in available raw prices
conditional on fixed masks. The baseline is itself linear. Exact channel
contributions and effective kernels are possible; modes need not be identifiable.

Sampling delta fixed0.25hours, removing A/delta scale confounding. Initialize
periods1/6/12/24hours for four modes,0.5/1/2/4/6/12/24/168hours for eight modes;
decay times twice initial period. These are declared initialization anchors,
not discovered cycles. Stable negative real parts; frequencies/decays learn.
Compare identical frozen-frequency control (decays/amplitudes/readout still
learn). Digital frequencies reported from angle(exp(delta*A)), with Nyquist
aliasing accounted for. Long modes may be weakly identified by a seven-day window.
Effective kernels and cross-fold consistency matter more than individual labels.

Approx1104/1264 total parameters, including eleven quantile heads; delta fixed.
Point readout80weights; uncertainty readout uses positive softplus increments,
so uncertainty is nonlinear while the point decision remains affine. State and
readout parameters can trade scale; export effective mode amplitudes/kernels.

## Unchanged validation

VALIDATION_PROTOCOL.md: eight February–September expanding folds; training
October2025 onward, previous-month1–15 selection,16–end calibration, frozen test
parameters/preprocessing, strictly prior observed quarters. Fit from scratch;
all days' zones/quarters stay together. Train-only affine scaling and empirical
quantile initialization. Two families x two state sizes x two rates(.001/.003)
x seeds17/43 =16candidates/fold,128fullfits. Maximum300epochs/patience35,batch32,
MAE(conservedq50)+mean eleven-quantile pinball; state learning rate0.1×,zero decay.
Select by inner q50MAE ratio only. No outer-test tuning or deployment replacement.

Primary arithmetic mean eight monthly q50MAE/mean-corrected-interpolationMAE
ratios; also pooledMAE, separate pooledratio, monthly/zone results, pinball,
50/80/90/95%CQRcoverage/width and relative-width/error association. Save all eleven
quantiles and intervals. Developmental backtests; no blind/significance claim.

First/last-fold all-family smoke; source hash/current-future perturbations;
independent calendar-history and query masks; all128checkpoint/selection replay;
independent calibration/score audit; exact superposition, sequential recurrence
and numpy convolution replay; exact source-channel contribution sum on first
held-out day of every family/fold. Export frequencies.csv,frequency plot, and
per-fold kernels/contributions for hours0/12/23. Frozen-frequency comparison
isolates frequency learning, with learnable decays shared across both arms.

Current execution-host Nightwatch wrapper reports success/failure. Completion
requires full accuracy and interpretability audits, then hashed recovery archive.
