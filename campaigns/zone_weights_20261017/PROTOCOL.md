# Posthoc zone-specific quantile ensemble weights

Authorized 17 October 2026 for the existing Vast.ai RTX 5090. Freeze this
protocol before evaluating its new test predictions. All eight chronological
folds, expert checkpoints, eleven quantile levels, official data constraints,
and interpolation denominators are inherited from the completed all-quantile
campaign. No expert is retrained and no production artifact is replaced.

For each expert-training objective (ordinary/uniform and month-ratio weighted),
reuse its six validation-selected experts. Fit nonnegative weights summing to
one by minimizing median MAE on preceding-month days 1–15. Compare:

- One global vector shared by all zones.
- Four separate zone vectors.
- Zone vectors shrunk toward the fitted global vector with zone fractions
  0.25, 0.50, and 0.75; fraction 0 is global and 1 is unrestricted per-zone.
- A common zone fraction selected using days 1–10 for preliminary weight fitting
  and days 11–15 for scoring the fixed grid 0/.25/.5/.75/1.
- Four zone fractions selected on that same chronological split.

Refit weight vectors on all 15 selection days after freezing the fractions.
An additional pre-test selector chooses the expert-training objective and
common/per-zone fraction policy using the days 11–15 score. These internal
scores are tuning scores: expert checkpoint selection already used days 1–15,
so this is not a fully nested or independent inner validation experiment.
The following full test month is never supplied to weight fitting or selection.
Only one selection month is fitted in each fold, so its common interpolation
denominator leaves the global MAE optimum unchanged. Per-zone MAE minimizers
also minimize that month's aggregate MAE because zone weights are independent.
Across test months report the arithmetic mean of monthly MAE/interpolation-MAE.

Use GPU full-batch objective/gradient evaluation with constrained SLSQP over
simplex weights, progressively smoothing absolute error down to 0.0001 EUR/MWh.
Report exact, unsmoothed fitting MAE. Check convergence, nonnegative simplex
constraints, fit improvement over initialization, and compare first-fold global
and all four zone solutions against independent exact linear programs. Do not
claim exact optimization from a smoothed objective alone.

Apply each weight vector unchanged to every hour, all four quarters, and all
eleven quantiles in a zone. This preserves the conserved median's hourly mean
and quantile ordering. Calibrate each final blend separately using preceding
days 16–end and the existing widen-only zone/quarter CQR procedure. Save all
quantiles, calibrated 50/80/90/95% intervals, and relative-width confidence
labels; assess pinball, coverage, width and absolute error by confidence group.

Compare matched existing global blends and their validation selector. Report
monthly and zone results and readable model weights. All results remain
developmental; no country-transfer claim is established. Portability means
reusing weight fitting and separate interval calibration with compatible frozen
experts and labeled local history, not assuming weights transfer unchanged.

Run on the remote execution host with current Nightwatch heartbeats and actual
completion/crash notification. Audit before success, archive the experiment,
and copy verified results off the ephemeral instance.
