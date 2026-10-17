# Champion submission bundle

This directory contains the reproducible refit campaign. The generated standalone bundle and fitted weights are written to `results/submission_champion_20261017/bundle/`; they are not the historical frozen `submission/` bundle.

Fit on the authorized research server from the project root:

```sh
python run.py campaigns/submission_champion_20261017/refit.py
```

The exported bundle includes `run.py`, `sources/`, `model.pt`, `recipe.json`, `calibration.json`, `SPECIFICATION_CARD.md`, pinned requirements and SHA256 manifest. Inside the exported folder:

```sh
python -m pip install -r requirements.txt
python run.py --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

Add `--quantiles-output uncertainty.csv` to save eleven raw quantiles and calibrated 50/80/90/95% bounds. Inference runs offline on CPU and does not fit or recalibrate. Only load the trusted supplied model artifact.

Hourly CSV columns: `delivery_date,market_id,hour_index,hourly_price_eur_mwh`. History columns: `delivery_date,market_id,hour_index,quarter_index,price_eur_mwh`. Dates use ISO YYYY-MM-DD; zones are ZONE_0–ZONE_3; hours 1–24 and quarters 1–4. Each input day needs 96 finite hourly values. History contains complete observed days strictly before the first requested day; missing historical dates are handled by availability features. For a batch of future days, future predictions are never inserted as observed history. To use newly observed earlier days, invoke again with updated history.

Output columns: `delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh`, with 384 rows per day. The four point predictions preserve each supplied hourly mean within 0.0001 EUR/MWh. Nonmedian quantiles and interval endpoints are marginal bounds and need not preserve that mean.

The campaign checks source identity and causal cache probes, fits six fresh fixed-budget networks on 348 dates, calibrates on 15 separate dates, checks checkpoint replay through the standalone CPU runtime, exercises the CSV contract, and packages an archive. `audit.json` and `completion.json` report actual results. Calibration-set coverage is not a future-test result. Packaging status must be read from those receipts; source preparation alone is not a completed fitted model.
