# Error minimization campaign — 2026-10-16

## Follow-up: weighted hourly-mean correction

The fifth campaign retests up to six selected/promising CatBoost configurations at their fixed fourth-campaign tree counts. Each raw four-quarter residual vector is corrected as `r - sum(r) * alpha`, with nonnegative four-element alpha summing to one per zone. Uniform alpha reproduces the original projection. Coefficients are fit on the first 30 validation dates and must improve the later 30 before being refit on all 60; otherwise uniform correction is retained for that zone. The fitting objective adds fixed 0.05 squared shrinkage towards uniform alpha. Both uniform and learned corrections remain ensemble alternatives. No holdout targets enter correction fitting or selection.

Numerical checks passed for zero hourly residual sums, equivalence of uniform corrections and isolation from perturbed holdout targets. The fifth full search completed with validation MAE 3.664430 and holdout 4.780831; it did not beat the fourth search's validation 3.663728 and was not selected. Its selected components passed offline eligibility QA. Snapshots are in campaigns/projection/. GPU1 was verified active during training under Nightwatch 76220b995fab9750c2f78305, which completed with exit zero.

The user questioned the 60-day validation allocation. It was a conservative starting choice to reduce sensitivity to volatile individual days, not a challenge requirement or an optimized protocol. A rolling 28-day comparison could use more history per fit and assess multiple periods. Existing results retain their original 243/60/60 split; the final fitted artifacts always use all 363 dates.

## Follow-up: GPU absolute-error boosting

A fourth campaign adds CatBoost 1.2.8, trained from random initialization on official data only. Standard-library use is permitted by the challenge; no pretrained weights or external prices are downloaded. CatBoost's official [regression loss documentation](https://catboost.ai/docs/en/concepts/loss-functions-regression) lists MAE as a GPU-supported training objective.

The library and graphviz 0.21 are installed in the isolated remote `catboost_vendor/` directory, without changing the shared research environment. Requirements for ordinary installation are in `requirements-catboost.txt`. `train_catboost.py` consumes the third search's checked candidate-prediction cache and frozen fitted ensemble as a parent candidate. It checks the source hash, dates, targets, and exact reproduction of the parent's validation MAE before adding candidates.

Nine pooled models append quarter position and its four indicators to the 389 rich hourly features: depth 6/8, learning rate 0.03/0.08, L2 penalty 3/30, plus depth 10 at 0.03/L2 30. Two further configurations fit four quarter-specific models, depth 6/8, learning rate 0.05 and L2 30. Maximum 1,800 trees, validation early stopping after 100 unimproved rounds. The pooled configurations use seed 17 except the depth-10 configuration (43). Predictions are mean-projected before scoring. No holdout target enters fitting or model selection.

Global and zone-specific convex weights combine these candidates with the validated parent ensemble and corrected interpolation. The zone objective retains fixed 0.05 shrinkage towards the shared weights. Selected new components are refit on all official dates for their fixed selected tree counts. Parent models were already refit on all dates and remain frozen. `predict.py` supports a nested parent ensemble and CatBoost components; its offline CSV QA must pass before success.

`run_catboost_after.py` waits for the third tracked wrapper to terminate, requires successful QA, runs a three-tree smoke test, then the full search. Remote sources are isolated in `catboost_campaign/`. The wrapper finally compares all four pipeline validation scores, updates `results/best_search.json`, and leaves a previous winner in place if the new family is worse. The fourth pipeline's model, metrics and heldout diagnostics will be under `results/catboost_search_gpu1/`. Reproduction requires the three prior training stages and this trainer; inference alone needs the chosen model and the dependency file. Repeated validation selection and previously inspected holdout diagnostics remain limitations.

## Follow-up: ensemble weights by zone

After both full searches completed successfully, the rich features improved validation MAE from 3.753788 to 3.676061 and holdout MAE from 5.036200 to 4.787278. The third campaign preserves the richer candidate family and split, adding `--zone-ensemble`: a separate convex mixture per zone with fixed penalty `0.05 * sum((zone_weights - shared_weights)^2)`. This shrinks the zone mixtures towards the shared mixture. Weights are selected only on validation dates, independently of holdout results. Small weights below 0.005 are removed and columns normalized. The union of selected components is refit on all supplied training dates.

`run_zone_search.py` runs a smoke check and full search sequentially on GPU1, then compares all three complete pipelines by validation MAE only. It writes `results/best_search.json`. Candidate predictions are now cached in `selection_predictions.npz` to enable later calibration experiments without retraining every candidate. Remote code is isolated in `zonal_campaign/`; local snapshots of the first two campaigns are in `campaigns/`. Every fitted model stores either a scalar weight or four zone weights; the offline entry point supports both.

## Follow-up: explicit slope and historical-shape features

The first search's gradient boosters improved on the smoke ensemble, reaching validation MAE 3.856099 individually. This motivated a second, otherwise matched search using `--rich`. It expands the 307 original features to 389: left/right hourly slopes across zones, slope magnitudes and curvature, daily level/range/quantiles, numeric hour/weekday/month/weekend, cross-zone spreads, historical interpolation errors at lags 1/2/7/14 days, and medians over the preceding 7/14/28/56 calendar days. A fixed nearest-curve rule also averages interpolation errors from the ten closest local hourly curves within the preceding 90 days. Distance uses five centred hourly observations plus price level. Its formula and weights are fixed in code; it does not fit model parameters during inference.

`verify_rich_features.py` passed on Ando: 192 rows, 307 basic and 389 rich features; finite values with empty history; exact agreement of original feature prefix; current/future target perturbations and removing unavailable history both leave the first evaluated day's features unchanged. The follow-up performs a full GPU smoke check before its full run. It is staged in `/home/mnogales/Projects/HACKATON/rich_campaign/` to preserve the first process's imports until that process completes.

`run_rich_after.py` checks the original training PID's command identity, waits for its exit, and requires successful offline QA before starting. Nightwatch tracks this durable queue and the follow-up. Both searches have the same chronological split. `results/best_search.json` will select between them using validation MAE only. Holdout scores are diagnostic and have already been viewed in smoke testing. This repeated use of validation dates increases selection optimism; secret-date performance remains unknown.

Reproduce the richer search directly with `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python train_search.py --rich --out results/rich_search_gpu1`. Offline `predict.py` reads the saved feature mode and also supports basic artifacts without that field.

Engineering smoke run: a short run exercised the same split, including producing a holdout diagnostic. The full search candidate list was written before that output was inspected and was not changed in response. The holdout is excluded from automated parameter/ensemble selection, but its smoke score has been viewed; it is not a never-viewed test set. Secret evaluation targets remain unavailable.

Objective: reduce the official equally weighted zone MAE while keeping each hour's four predicted quarters at the given hourly mean. Only the supplied official training CSV is used. No external data, pretrained models, or October 2026 prices are used.

The 363 dates are sorted chronologically. The first 243 dates fit candidate parameters; the next 60 select configurations, neural stopping epochs, and convex ensemble weights; the last 60 form a holdout whose targets do not influence selection. Holdout scores are reported only after selection. Prior holdout days' observations can feed fixed history features on later holdout dates, matching the challenge's sequential history provision. This is not an isolated forecast of all 60 days without new history. All four zones stay together in each split.

Features use all four zones' available daily hourly curves, nearby hourly prices and differences, hour/zone/weekday indicators, cyclic calendar features, and within-hour historical residuals at calendar lags 1, 2, and 7 days plus trailing 7/28-calendar-day means. Missing historical lags become zero. History is always filtered strictly before the reconstructed date. A perturbation assertion verifies that modifying current and future quarter-hour targets leaves that date's features unchanged. DST gaps are not treated as ordinary adjacent calendar days.

Candidates: compliant centred interpolation, flat hourly repetition, five ridge penalties, four ExtraTrees leaf sizes, four absolute-error gradient boosters, and sixteen randomly initialized neural networks with varying width/depth/dropout/learning rate/seed. Neural networks optimize absolute error with projected zero-mean residual outputs. Early stopping uses only validation dates. The ensemble has nonnegative weights summing to one; weights below 0.005 are removed and the remainder normalized. All outputs are explicitly mean-corrected.

The search saves candidate validation scores, a selection/holdout report, heldout arrays, and a fitted ensemble. Selected components are then retrained on all 363 permitted dates with the selected settings and neural epochs fixed. The final refit has no independent test score; the heldout score pertains to models fit on the first 243 dates. One temporal holdout cannot establish secret-test performance, especially across seasons.

Run training on Ando GPU 1 from `/home/mnogales/Projects/HACKATON`:

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/mnogales/miniconda3/envs/predictcsl-main/bin/python train_search.py
```

Offline inference (CPU supported; uses trusted locally generated pickle artifacts):

```sh
python predict.py --model results/search_gpu1/model.pkl --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

The inference entry point requires a complete 24-hour curve for each of the four zones and history preceding the earliest evaluated date. It writes the challenge's required five columns at ten-decimal precision. Training verifies the resulting CSV's uniqueness, finite values, four quarters per hour, and hourly-mean tolerance. Full search results will be under `results/search_gpu1/` on Ando; smoke results are separate and are not optimized performance claims. Dependency versions must match the verified execution environment; see `requirements.txt`.
