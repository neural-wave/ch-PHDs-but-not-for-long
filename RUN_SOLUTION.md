# Training and offline prediction

**Required validation for every future model comparison:** follow [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md). Fit from scratch on increasingly wider histories (3–10 months), select on the preceding month's days 1–15, calibrate on days 16–end, and test on eight full February–September months. Rank by the arithmetic mean of monthly MAE divided by mean-corrected linear-interpolation MAE; lower is better. Save eleven quantiles and preserve the supplied hourly means.

The current research champion is registered in [best_model.json](best_model.json), explained in [BEST_MODEL.md](BEST_MODEL.md), and compared with all recorded studies in [MODEL_RANKING.md](MODEL_RANKING.md). Rebuild the ranking from saved audited scores with `python3 rank_models.py`.

The existing offline inference bundle is the earlier frozen pipeline refitted on all 363 official days, with no held-out accuracy claim for that refit. Its explicit manifest is `results/deployment_best.json`; the newly promoted research recipe has no all-history refit yet. September-only commands in MONTHLY_PROTOCOL.md and the 60-day searches below are historical reproduction instructions.

Run commands from this project root with the supplied `train_prices_15min.csv` intact. Training uses only that official file. No external data or pretrained models are required. `README.txt` is the original official package description.

The validation set is 60 complete delivery dates, **3 June–1 August 2026**, four zones and 96 quarters per zone/day: **23,040 quarter-hour targets**, grouped into 5,760 hours. The chronological training set has 243 days (1 October 2025–2 June 2026, with DST dates excluded); the diagnostic holdout has 60 days (2 August–30 September 2026). All zones from a delivery date remain in one split. Earlier holdout observations may feed fixed history features on later holdout dates, matching the challenge's sequential history provision. Parameters are not updated during inference.

Validation selects candidate settings, early-stopping epochs, ensemble weights, and correction weights. It is a development set whose score is optimistic after repeated selection. Holdout diagnostics have also been inspected during development; neither set is a substitute for the unavailable secret test.

Install the pinned dependencies in a dedicated environment:

```sh
python -m pip install -r requirements-catboost.txt
```

The verified training environment on Ando is `/home/mnogales/miniconda3/envs/predictcsl-main/bin/python`; CatBoost is isolated in `/home/mnogales/Projects/HACKATON/catboost_vendor`. When using that environment, add the vendor directory to `PYTHONPATH`. The shared environment was not changed. Use a compatible CUDA build of PyTorch for GPU training; offline prediction can run on CPU.

Reproduce the successive searches on GPU1, in order:

```sh
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python campaigns/basic/train_search.py --out results/search_gpu1
python campaigns/rich/train_search.py --rich --out results/rich_search_gpu1
python campaigns/zonal/train_search.py --rich --zone-ensemble --out results/zone_search_gpu1
python campaigns/catboost/train_catboost.py --out results/catboost_search_gpu1
python train_catboost.py --projection-refine --out results/projection_search_gpu1
python select_best.py
```

The first three stages each search 31 candidates including baselines. The fourth searches eleven boosting configurations. The last stage retests up to six selected/promising configurations with frozen per-zone correction coefficients, retaining uniform corrections as alternatives. CatBoost GPU training can have small numerical variation between runs. Prior stages' caches and fitted artifacts are required by the following stages; all can be regenerated in this sequence.

The operational `run_*_after.py` launchers contain pointers to this campaign's historical processes; use the direct commands above to reproduce training. For a new unattended training campaign, register the command with Nightwatch on the execution host. The manifest is written only from complete models with successful inference checks; it chooses the lowest validation MAE and never reads holdout MAE for selection. Freeze the selected code, dependencies, model and manifest together before secret evaluation. Run inference with jury-provided earlier history and hourly curves:

```sh
python predict_best.py --selection results/deployment_best.json --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

The history file must precede the earliest evaluated delivery date. This entry point performs no training, recalibration, model selection or network lookup. It writes `delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh` and enforces four finite quarters per hour with the supplied mean within 0.0001 EUR/MWh. Separate test dates can be processed with their respective provided histories. Existing output eligibility checks cover the exported decimal CSV as well as in-memory predictions.
