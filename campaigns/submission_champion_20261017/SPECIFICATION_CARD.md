# Neural Wave champion — specification card

**Deployment recipe:** `champion_348fit_15cal_20261017`.
**Research parent:** `minimize_expanding_20261017/fixed_blend_ratio`.
**Status:** refit and packaging requested; completion requires the campaign's passing audit and completion receipt. This card describes the frozen recipe, not a claim that training has already finished.

## Purpose and inputs

Reconstruct four quarter-hour electricity prices per supplied hourly mean for four anonymized zones. Uses official challenge data only. The complete current-day four-zone hourly curve is known input. Observed quarter history is filtered to strictly earlier dates. No current-day observed quarters, external data or pretrained weights are used.

## Architecture and output

Mean-corrected linear interpolation supplies the anchor. Each network learns zero-mean within-hour residuals; convex averaging preserves the q50 hourly mean. Eleven ordered marginal quantiles: 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.975. Only q50 supplies the submitted point decision and has an hourly conservation guarantee. Quantile-function averaging does not produce the quantiles of a mixture distribution.

| Family | Ensemble weight | Architecture | Seeds and fixed epochs |
|---|---:|---|---|
| Compact MLP | 1/3 | Two hidden layers, 32 units, SiLU; 194 engineered inputs | 17:112, 43:134 |
| Rich temporal CNN | 1/3 | Two 16-channel convolutions, five-hour kernels, SiLU | 17:54, 43:33 |
| Analog 2D CNN | 1/3 | Joint-zone historical grid, 16-channel convolution/fusion and full hourly curve | 17:199, 43:176 |

Each seed has total weight 1/6. Learning rates: MLP and temporal CNN 0.003; analog CNN 0.001. Fixed physical feature scaling and clipping to [-15,15]; no RevIN. AdamW, weight decay 0.01, batch size 32, gradient clipping 5. Joint conserved-median MAE and quantile pinball objective, with fitting-month interpolation-normalized weights. Exact configuration and source hashes: `recipe.json`.

“Analog” features retrieve up to ten similar earlier days from the preceding 90 calendar days, matching local hourly-curve shape and price level per zone/hour. Their inverse-distance-weighted historical interpolation residuals provide four causal quarter-correction channels. This is an input feature to the neural model.

## Fitting and calibration

Fresh fitting: October 1, 2025–September 15, 2026, **348 eligible dates**. Calibration only: September 16–30, 2026, **15 dates**. The user approved this holdout in place of fitting all 363 dates. Architectures, learning rates, seeds, weights and component epoch budgets come from the latest research fold, whose selection block was August 1–15. No early stopping, feature fitting or model selection uses calibration labels. Earlier observed calibration days may inform later calibration-day causal history, matching the rolling-observation assumption.

Calibration uses widen-only finite-sample conformalized quantile residual corrections, separately by zone and quarter position, pooling day/hour observations, followed by deterministic nesting for 50/80/90/95% intervals. Temporal dependence and distribution shift prevent a general future-coverage guarantee. These September dates were inspected during research, so this is parameter-held-out calibration, not an untouched final benchmark.

## Research evidence and limits

Eight expanding developmental February–September folds: equal-month MAE/interpolation-MAE **0.707968**, a **29.20% reduction**; pooled MAE **4.103184 EUR/MWh** on 92,544 quarter targets. Beat interpolation in all eight months. Research calibrated 90% coverage **90.59%**, mean total width **18.8354 EUR/MWh**; monthly coverage ranged **85.83–93.16%**. These results belong to the research procedure and are not measured accuracy or coverage of this new deployment refit. The historical champion used different selected blends in February/March; this deployment freezes the latest three-family recipe.

The recipe was promoted after comparing developmental campaign results. Months are dependent and September is not blind. Marginal coverage is the fraction of individual observed quarters inside their bounds; it is not simultaneous-hour/day coverage, confidence in the model, or a probability that a point prediction is correct. Extreme quarter spikes can remain difficult despite a known hourly mean.

## Interface, verification and artifacts

Offline CPU inference: `python run.py --hourly INPUT --history HISTORY --output OUTPUT`; optional `--quantiles-output UNCERTAINTY`. See `README.md` for schemas and history availability rules. Requires Python compatible with pinned NumPy, pandas and PyTorch versions; fitted bundle records its actual environment. No GPU is needed for jury inference.

Required evidence: official-data/source hashes, causal perturbation/truncation checks, fresh fitting and calibration split, finite ordered quantiles, isolated CPU checkpoint replay, CSV key/row identity and hourly-mean error <=0.0001 EUR/MWh. The generated `audit.json` reports each result and clean-environment status. The generated SHA256 manifest and archive receipt identify the deliverable. The existing frozen submission bundle remains a separate historical artifact until deliberate audited replacement.
