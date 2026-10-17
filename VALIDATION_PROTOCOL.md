# Monthly chronological validation

**Standing user instruction, reaffirmed 17 October 2026:** use this validation
method for every future model comparison and research-model promotion. The
primary ranking must use these eight expanding folds and the equal-month
MAE/mean-corrected-interpolation MAE score. Preserve fresh fitting, prior-only
quarter history, selection/calibration separation, frozen test parameters and
the default eleven quantiles. Keep other protocols as clearly labelled
diagnostics; they cannot replace missing evidence in the primary ranking.
Current champion and consolidated scores: `BEST_MODEL.md`, `best_model.json`
and `MODEL_RANKING.md`.

User update,17October2026: quantiles are the default for every subsequent predictive-model comparison. Save eleven ordered marginal quantiles at0.025/0.05/0.10/0.20/0.25/0.50/0.75/0.80/0.90/0.95/0.975 and calibrated50/80/90/95%intervals. The conserved q50 decision supplies the point MAE ratio. Fit calibration on the reserved16–end block only. Report pinball loss, interval width/empirical coverage and whether narrower intervals correspond to smaller held-out errors. Preserve per-quarter quantiles and interval data for later inspection. Width labels are relative uncertainty measures, not probabilities of prediction correctness. See `campaigns/monthly_quantiles_20261017/PROTOCOL.md`; earlier point-only results remain historical controls. This instruction reopens quantile uncertainty inspection; the previously paused catastrophic-error/risk-classifier branch remains set aside.

Adopted 17 October 2026 at the user's request. This formalizes the existing cross-validation-like monthly backtest and adds error normalization by the same-month interpolation baseline. It supersedes raw-MAE-only reporting as the primary **cross-month comparison**, while retaining raw MAE for the challenge metric. Historical artifacts and deployment selection remain intact.

## Folds and selection

Use eight expanding folds, starting with three training months. Training always starts in October 2025. Every fold fits from scratch using only its training dates, selects models using days 1–15 of the immediately following month, optionally calibrates intervals using days 16–month-end, then tests on the next full month.

| Fold | Training months | Selection/calibration month | Test month |
|---|---|---|---|
| 1 | October–December 2025 | January 2026 | February 2026 |
| 2 | October 2025–January 2026 | February 2026 | March 2026 |
| 3 | October 2025–February 2026 | March 2026 | April 2026 |
| 4 | October 2025–March 2026 | April 2026 | May 2026 |
| 5 | October 2025–April 2026 | May 2026 | June 2026 |
| 6 | October 2025–May 2026 | June 2026 | July 2026 |
| 7 | October 2025–June 2026 | July 2026 | August 2026 |
| 8 | October 2025–July 2026 | August 2026 | September 2026 |

Keep all zones/hours/quarters of a date together. Fit preprocessing, normalization, additive knots and parameters on that fold's training data only. Select features, candidate configuration, epoch, ensemble weights and any training-window choice using pre-test selection data only. Freeze parameters and calibration through the test month. Do not refit on selection/calibration dates in this protocol. Earlier test dates may enter later folds' training, following chronology. Exclude the official package's missing DST dates consistently. February–September contains 241 eligible days and 92,544 quarter targets.

This is monthly walk-forward validation with one chronological inner selection block. Full nested rolling inner selection is a separate, more expensive protocol and is not claimed for these saved results. Freeze each new campaign's candidate list, selection objective and fit budget before looking at its test outcomes. The existing linear-quantile campaign uses a joint MAE/pinball selection objective; retain that provenance when comparing it with MAE-selected pipelines.

## Inputs and constraints

Official challenge data only for eligible methods. Today's complete four-zone hourly curve is known. Observed quarter-hour history must be strictly earlier than the reconstructed day and available under deployment rules; later test days may use earlier observed test days, with no parameter updates. Every predicted hour must contain four finite quarters with mean matching the supplied hour within 0.0001 EUR/MWh. Keep external/pretrained exploratory models separate from eligible comparisons.

## Scores

For model j and test month m, compute quarter MAE on the identical dates, zones and quarters for model and baseline:

`R[j,m] = MAE[j,m] / interpolation_MAE[m]`.

The primary denominator is **mean-corrected linear interpolation**, the established feasible interpolation control: interpolate between hourly centres to quarter centres with flat extension at the day's boundaries, then add the same constant to each quarter within an hour so its mean equals the supplied hourly value. This definition is fixed across months and folds. Also record raw uncorrected linear-interpolation ratios under an explicit alternate name; never substitute one denominator silently. A zero denominator makes a ratio undefined and must be flagged rather than smoothed with an arbitrary epsilon.

Primary cross-month score: **arithmetic mean of the eight monthly ratios**, giving each test month equal weight. Ratio 1 means parity, 0.75 means 25% less MAE than interpolation, and above 1 means worse. Also report each monthly ratio, worst-month ratio, per-zone monthly MAE and same-zone ratio, and target-weighted pooled raw MAE in EUR/MWh. Keep `pooled model MAE / pooled interpolation MAE` as a distinct secondary statistic: it gives more weight to months with higher interpolation error. Compute ratios after aggregating absolute errors within each month, not by averaging quarter-wise or daily error ratios.

Ratios describe relative difficulty and performance using held-out targets. They are evaluation quantities, not inference-time inputs or an assumed available month-difficulty forecast. Within one selection month, its common denominator leaves the MAE ranking unchanged. If multiple inner months are used in a future campaign, select using their mean monthly ratios rather than any outer test result.

The fixed-September training-history study is a separate diagnostic. Its eight points all reuse September and must never be counted as eight independent test folds. September-only CNN/ensemble campaigns stay in a separate table until they have matched multi-month results.

## Interpretation and reproducibility

All available months have already been inspected during development. Call these developmental backtests; resampling does not create a blind benchmark. Monthly folds share training history and are dependent. Do not derive independent-fold confidence intervals or claim that normalization removes all regime effects. No new production model is chosen from the test-month winner.

Quick evaluation: run `python3 evaluate_normalized_validation.py`. Saved scores, official-baseline recomputation, source hashes, audit receipts and monthly/aggregate tables are in `results/normalized_validation_20261017/`. Earlier execution protocols remain in `EXPANDING_PROTOCOL.md`, `EXPANDING_SIMPLE_PROTOCOL.md` and `LINEAR_QUANTILES_PROTOCOL.md`.
