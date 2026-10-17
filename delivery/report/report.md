# Quarter hour electricity prices with conserved hourly means

Neural Wave 2026 | AET challenge | Report draft, 17 October 2026

Team full names and optional roles: to be completed later.

## Goal and approach

An hourly price fixes the average of four quarters, but leaves their individual values unknown. Our goal is to recover that within-hour shape for four anonymized electricity-market zones. The complete daily hourly curves are supplied together. We use only the official package and quarter-hour observations from strictly earlier days. The official training data contains 363 eligible days from October 2025 to September 2026, excluding the two daylight-saving transitions.

We start from mean-corrected linear interpolation: interpolate between hourly centres, extend the endpoints, then shift each group of four quarters to match its supplied hourly mean. Learned models predict corrections around this feasible baseline. For an hourly mean H and preliminary quarter predictions u, the final prediction is p(q) = u(q) - mean(u) + H. This projection preserves the hourly mean without depending on the true quarter-hour targets.

The research ensemble compares rich and compact linear models, an additive model, an MLP, a temporal CNN and an analog 2D CNN. Inputs describe known hourly curves, calendar information and causal historical patterns. Fixed physical scaling avoids fitting a new feature standardization for each regime; additive knots use fitting data only. Neural optimization combines median absolute error and eleven-quantile pinball loss, using AdamW and selection-based stopping. The ratio-trained arm weights fitting months using their interpolation errors and counts, never a test-month denominator.

Candidate models and convex blends are selected using only the preceding month's first 15 days. April-September folds use equal MLP, temporal-CNN and analog-CNN weights. February blends rich linear and MLP; March averages six experts. Configurations, optional seed averages and weights remain fixed through each test month. This adds flexibility while retaining a simple final averaging rule.

## Validation and implementation

We use eight expanding chronological folds. Training starts with October-December 2025 and grows to October 2025-July 2026. The next month supplies selection on days 1-15 and separate interval calibration on days 16-end; the following full month is tested. Models fit from scratch in every fold. The evaluated February-September period contains 241 eligible days and 92,544 targets.

Python, NumPy, pandas, SciPy, scikit-learn and PyTorch support the research implementation. The main score is the equal-month mean of model MAE divided by same-month mean-corrected-interpolation MAE. Raw MAE remains in EUR/MWh. The official jury instead scores its secret dates with equal zone weights; our normalized research score is a development comparison.

<!-- pagebreak -->

## Results on developmental backtests

The strongest observed research recipe reached a mean monthly error ratio of **0.707968**, a **29.20% reduction** relative to mean-corrected interpolation. Pooled MAE was **4.103184 EUR/MWh**. The worst monthly ratio was **0.754293**. Every test month was better than this interpolation control. The pooled MAE ratio and the mean of monthly ratios are different statistics.

<!-- monthly_chart -->

| Evaluated recipe | Mean monthly MAE ratio | Pooled MAE, EUR/MWh |
| --- | --- | --- |
| Mean-corrected interpolation | 1.000000 | 5.814817 |
| Compact linear, fixed scaling | 0.724553 | 4.201780 |
| Original research champion | 0.707968 | 4.103184 |
| Champion with RevIN | 0.725133 | 4.206289 |

The champion predicts eleven ordered marginal quantiles, from 0.025 to 0.975. Reserved calibration dates produce nested 50%, 80%, 90% and 95% intervals. Its calibrated 90% interval covered **90.59%** of the evaluated targets, with mean width **18.8354 EUR/MWh**. Only the median point decision is constrained to conserve the hourly mean; marginal quantiles describe uncertainty and need not individually conserve it.

The campaign audit recomputed 384 candidate predictions and replayed 281 selected neural models with zero recorded replay error. Maximum hourly-mean error was **1.14e-13 EUR/MWh**, below the 0.0001 tolerance. Audits also checked causal features, fitting-only preprocessing, selection choices, quantile ordering and calibration.

All available months, including September, were inspected during development. The champion was chosen after aggregate developmental comparisons. It is not a blind benchmark result or an estimate of secret-date accuracy. A separate policy selecting among recipes from prior data reached **0.708371**; choosing that policy across campaigns is also developmental.

<!-- pagebreak -->

## Challenges and ideas that did not improve the solution

Preserving the supplied hourly means is an eligibility requirement. Direct quarter predictions can drift, so the final projection enforces the constraint algebraically. Historical features can also leak targets if a nearby day is admitted incorrectly; the pipeline restricts observed quarters to strictly earlier dates and the campaign audits test feature causality.

Normalization was not uniformly helpful. In later matched trials, adding reversible instance normalization (RevIN) increased the champion's mean monthly ratio from 0.707968 to 0.725133. The compact linear ratio increased from 0.724553 to 0.733498, and the matched compact CNN from 0.768726 to 0.787509. Intervals became wider and aggregate coverage rose, but point accuracy worsened. We retained the original research champion. A wider interval alone does not demonstrate a better model.

The full year remains small for learning price regimes, and repeated development comparisons can be optimistic. Expanding chronological folds preserve the deployment information flow, while the selection/calibration split avoids calibrating on the same dates used to choose a model. Those precautions do not turn inspected months into unseen evidence.

## Deployment and reproducibility

The research champion still needs a fresh final fit on all 363 official days. The currently preserved executable bundle belongs to an older recipe. Its all-history training does not establish held-out accuracy for that fitted artifact, and its earlier September result must not be attributed to the new champion. Calibration transferred to an all-history fit also needs a chronological audit before claiming coverage.

Final delivery must include matching training code, frozen configuration, dependencies, fitted artifacts and an offline inference entry point. The jury supplies hourly prices and earlier history; output contains four finite predictions per hour with complete unique keys and mean error <=0.0001 EUR/MWh. Clean-environment verification and delivery of the fitted artifact remain pending. Large fitted binaries are excluded from ordinary Git, so repository publication alone is insufficient.

## AI assistant usage and credits

OpenAI Codex assisted with code and analysis work, repository organization, documentation, and this report and video script. Reported numbers come from saved experiment evaluations and audits. Team review and the final member list remain pending; the disclosure should be checked against all assistants actually used. No external price data or pretrained forecasting weights are part of the eligible comparisons described here.

Data: official AET Neural Wave 2026 package, processed and anonymized by AET from ENTSO-E prices. Software: Python scientific libraries and PyTorch; CatBoost also supports the preserved older bundle. Repository evidence: `VALIDATION_PROTOCOL.md`, `best_model.json`, `results/minimize_expanding_20261017/{summary.csv,monthly_scores.csv,audit.json}`, and the champion/linear/compact-CNN RevIN studies. Final delivery status and commands: `delivery/README.md` and `delivery/RUN_FOR_JURY.md`.
