# Learned reconstruction results — 16 October 2026

Historical 60-day evaluation. For the current champion and the required eight-fold expanding validation, see [BEST_MODEL.md](../BEST_MODEL.md), [MODEL_RANKING.md](../MODEL_RANKING.md) and [VALIDATION_PROTOCOL.md](../VALIDATION_PROTOCOL.md). The scores and model choices below retain their original evaluation scope.

All numbers are EUR/MWh MAE, equally weighting the four zones. Every method below preserves hourly means. Validation: 60 days following 243 training days. Diagnostic holdout: final 60 official training dates. Prior days' quarter-hour history is available sequentially; no same-day quarter-hour targets are used as inputs.

| Method | Validation MAE | Holdout MAE |
|---|---:|---:|
| Corrected interpolation | 5.572185 | 6.682586 |
| Short smoke ensemble | 3.942620 | 5.109000 |
| Full basic-feature ensemble | 3.753788 | 5.036200 |
| Full richer-feature ensemble | 3.676061 | 4.787278 |
| Zone-specific ensemble | 3.664742 | 4.781314 |
| GPU MAE boosting expansion | **3.663728** | **4.779861** |
| Learned hourly-mean corrections | 3.664430 | 4.780831 |

The validation-selected boosting expansion reduces holdout MAE by 28.47% against corrected interpolation. Most of the gain comes from the original learned models and richer features. Boosting's additional holdout gain over the zone ensemble is only 0.001452 EUR/MWh; its paired seven-day block-bootstrap 95% interval includes zero (-0.000853 to 0.003992). These are descriptive diagnostics from 60 already-viewed days, not evidence of improved secret-test performance. See paired_diagnostics.json. The correction refinement did not beat the existing validation winner and was not selected. Repeated validation selection introduces optimism; the final all-data refit has no independent performance measurement.

Holdout MAE by zone for the richer ensemble: ZONE_0 3.578789, ZONE_1 5.933715, ZONE_2 5.244459, ZONE_3 4.392148. Maximum hourly-mean error is 1.14e-13 EUR/MWh. Both completed searches passed exported offline CSV checks for 384 unique finite predictions, four quarters per hour, and the 0.0001 mean tolerance.

Metrics, heldout arrays and final all-data fitted models from the first four searches are retrieved locally and checksums match Ando. The fifth search's metrics and heldout arrays are retrieved; its losing fitted artifact remains preserved on Ando. The selected model is `results/catboost_search_gpu1/model.pkl`. Offline command from the project root, after installing requirements-catboost.txt:

```sh
python predict_best.py --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

All five searches have completed with offline eligibility checks passing. Code snapshots are under campaigns/. The manifest results/best_search.json chooses the lowest validation MAE among the five complete pipelines; no holdout score is used in that choice. RUN_SOLUTION.md provides reproduction and offline entry-point commands. Current validation size is 60 dates (3 June–1 August 2026), or 23,040 quarter-hour targets. The user questioned this conservative allocation; a rolling 28-day comparison is the next protocol improvement to assess. No revised split has yet been applied to the numbers above.
