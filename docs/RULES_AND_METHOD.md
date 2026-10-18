# Rules and method

The supplied [challenge PDF](challenge_rules.pdf), SHA-256
`986040c07a082de1089ed0fa4f0b5488d0a6ccd9c2af3c3b412a816199821d78`,
is the reference for this package. Relevant requirements are in sections 9, 11
and 12. This checklist describes the implementation, not a jury eligibility
decision or a claim about the unavailable original model binary.

| Rule | Implementation/evidence |
| --- | --- |
| Separate training and inference entry points | `src/train_ensemble.py` and `src/predict_prices.py` |
| Regenerate artifacts from official input | CSV hash check, direct causal feature generation, six fresh fits, held-out calibration, single checkpoint |
| Explicit dependencies, seeds and commands | README, requirements files, unchanged final recipe, per-run environment and checksum JSON |
| Only official development data | Copied 2025-10-01–2026-09-30 CSV; full training rejects other hashes |
| No external/pretrained weights | Networks initialize from recorded seeds; no downloads in training or inference |
| Known full-day hourly curve and prior observed quarters only | Calendar-date filters in all history features; query-day/future target poisoning and truncation probes |
| Frozen inference after submission | Inference loads weights/corrections and computes fixed features; no optimizer or calibration fitting |
| Inference without network or code edits | Documented explicit CSV CLI; isolated trial disables socket connections |
| Complete finite five-column CSV; conserved means | Read/write checks plus independent trial validator; maximum mean error threshold 0.0001 EUR/MWh |
| Jury may retrain from scratch | Full 348-day fit / 15-day calibration command with fixed original epochs; full retrain is not run as part of the small trial |

## Model assumptions

The hourly mean supplies the most reliable level. The model learns corrections
to mean-conserving linear interpolation, using daily shape, calendar structure,
cross-zone relationships and recurring quarter-hour patterns from earlier
observed days. The compact MLP and temporal CNN express local corrections; the
2D CNN jointly processes all zones and paired quarter/hour rows for the previous
seven calendar days and the current hourly curve. Availability masks preserve
the distinction between missing history and observed zero prices. A fixed
nearest-analogue rule retrieves up to ten earlier patterns from a 90-day window.
It is a fixed inference feature rule, not inference-time model fitting.

The selected models use fixed physical scales and clipping, without RevIN.
Each network produces an hourly mean-conserving median and eleven ordered
quantiles. Their equal-weight mixture preserves both properties. Widen-only
interval corrections are estimated once on the reserved calibration block,
pooled over days and hours separately for each zone and quarter; the resulting
50/80/90/95% intervals are nested. Dependence and regime changes mean these
intervals should not be read as a guarantee of future empirical coverage.

## Developmental validation and data limitations

Research comparisons used eight chronological expanding trials. Training
starts in October 2025 and grows each trial; the following month's days 1–15
select the configuration and days 16–end calibrate uncertainty. The next whole
month is tested with model and interval parameters frozen. Test months span
February through September 2026. Keeping one full future month per trial reduces
reliance on an unusually difficult month. Only strictly earlier observations may
enter a test date's history features.

The primary comparison is the arithmetic mean of the eight monthly ratios
MAE / mean-corrected-linear-interpolation MAE, so each month has equal weight.
The saved research champion ratio is approximately 0.708. The original registry's
pooled raw MAE, approximately 4.103 EUR/MWh, weights monthly MAEs by their number
of eligible dates; it is distinct from the unweighted average of monthly MAEs
and from the primary mean of ratios. None of these development scores is
measured by the tiny execution trial. The organizer's official point metric is
raw MAE with equal zone weights, and its official baseline repeats hourly means;
our interpolation ratio is an additional research comparison.

Data cover one year and omit the two DST dates. Negative/near-zero prices,
spikes and changing seasonal regimes make the inverse problem ambiguous: four
quarters cannot be uniquely recovered from one mean. All available research
months were inspected during development. The expanding trials share training
history and are developmental evidence, not independent or blind confirmation.
The final deployment fit uses the separate fixed 348/15 split in the README;
reproducing that fit does not rerun the eight-fold model search.
