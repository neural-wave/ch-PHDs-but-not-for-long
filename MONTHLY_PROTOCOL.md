# Requested monthly validation and test protocol

The user's instruction supersedes the rolling-window proposal and the earlier 60-day split.

| Purpose | Delivery dates | Days | Quarter-hour targets |
|---|---|---:|---:|
| Fit model parameters | 1 October 2025–31 July 2026 | 302 | 115,968 |
| Validation/model selection | 1–31 August 2026 | 31 | 11,904 |
| Development test | 1–30 September 2026 | 30 | 11,520 |

The two DST dates are excluded from the official training package. All four zones stay together within each date's split. Model parameters, normalization statistics and gradient updates use only the first 302 days. August chooses stopping epochs, configurations and ensemble weights. September never chooses a configuration or weight in this campaign. It was inspected in previous development, so this test is not a new blind test. Baseline and model scores must be compared on the same September dates; the old 60-day scores use different periods.

Each shared base predictor receives all four zones' 24 hourly prices together: 96 current-day hourly values, plus engineered context. The richer feature vector has 389 entries, including neighbouring hourly values, cross-zone differences, calendar fields and strictly earlier historical quarter-hour shapes. Base models predict four quarter residuals for one zone/hour at a time and enforce the given hourly mean. The base neural models are MLPs; other candidates are ridge regressions and tree ensembles. A subsequent joint-node comparison processes all four zones and 24 hours together using temporal convolutions and learned fully connected node attention, under the same monthly split.

The fixed history rule may use earlier August or September observations on later dates, matching the challenge's jury-provided history. It performs no parameter fitting or recalibration during inference. Current-day quarter targets are never inputs. Every output hour contains four finite predictions preserving its supplied mean within 0.0001 EUR/MWh.

Train the base family, then the GPU boosting comparison, on Ando GPU1:

```sh
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python train_search.py --rich --zone-ensemble --calendar-split --defer-refit --out results/monthly_base_gpu1
python train_catboost.py --parent-dir results/monthly_base_gpu1 --defer-refit --out results/monthly_cat_gpu1
python select_monthly.py --exclude-graph
python train_graph.py --out results/monthly_graph_gpu1
python select_monthly.py
```

The installed Ando environment uses the isolated `catboost_vendor` directory via PYTHONPATH; an ordinary dedicated environment can install requirements-catboost.txt. The base search has 31 candidates; the boosting comparison adds eleven configurations. `run_monthly.py` runs smoke checks and both full stages, verifies split counts/302-day fitting/offline CSV eligibility, then chooses the lower August MAE. It writes results/monthly_best.json, without comparing scores against the old validation period.

`--defer-refit` preserves the fitted 302-day models for test evaluation and inference. It prevents the old automatic all-363-day production refit. The chosen artifact, normalization and mixture are frozen from training/August selection. A later production refit can be performed explicitly after evaluation; no September-fitted model will be presented as the model that obtained the September test score.

When the tracked run completes, offline prediction is:

```sh
python predict_best.py --selection results/monthly_best.json --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

Historical manifests and snapshots remain available. The discarded rolling smoke run 7414cb7764019baafdd79a4f crashed before the full search: its reduced three-model subset had zero anchor weight in some zones. The normalization bug has been fixed, and that run is superseded by this monthly campaign rather than restarted.
