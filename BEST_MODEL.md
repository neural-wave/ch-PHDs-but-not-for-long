# Current best research model

Promoted at the user's request on 17 October 2026: **`minimize_expanding_20261017/fixed_blend_ratio`**.

Mean monthly MAE/interpolation MAE: **0.707968** (**29.20% reduction**). Pooled MAE: **4.103184 EUR/MWh**.

The quantile ensemble learns zero-mean within-hour corrections around mean-corrected linear interpolation. It uses known four-zone hourly curves and strictly prior quarter history, fixed physical feature scales and fitting-only additive knots. Models fit from scratch as training expands from three to ten months. Preceding-month days 1–15 select models, optional two-seed averages and global convex blend weights; those stay fixed throughout the following test month. Days 16–end separately calibrate all eleven quantiles and 50/80/90/95% intervals.

April–September uses equal MLP/temporal-CNN/analog-2D-CNN weights. February uses 25% rich linear and 75% MLP; March averages all six experts. The ratio objective weights fitting months by their own interpolation errors and counts; no test denominator is an input.

Calibrated 90% empirical coverage: **90.59%**; mean width **18.8354 EUR/MWh**.

The best observed recipe was identified after reporting the developmental test aggregates. The strongest separate prior-data selection policy is `ensemble_normalization_20261017/validation_selector` with ratio **0.708371**. These are distinct claims; cross-campaign rankings remain developmental.

Canonical registry: [best_model.json](best_model.json). Full ranking: [MODEL_RANKING.md](MODEL_RANKING.md). Exact per-fold configurations, weights and checkpoint hashes: [champion_per_fold.json](results/model_ranking_20261017/champion_per_fold.json). Audit: [campaign audit](results/minimize_expanding_20261017/audit.json). Artifacts: [verified recovery archive](results/minimize_expanding_20261017_completed.tar.gz), [checksum receipt](results/minimize_expanding_20261017/backup_verification.json).

This registry promotes the research recipe. The existing 363-day inference bundle still belongs to the earlier frozen pipeline. The new recipe has no all-history deployment refit; the registry does not label an evaluation checkpoint as such a refit.

Every future comparison and promotion must follow [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md).
