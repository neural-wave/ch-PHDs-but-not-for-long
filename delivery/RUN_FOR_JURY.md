# Running the AET solution

**Final-ensemble entry point:** use the [standalone jury pipeline](../jury_pipeline/README.md) for the final six-network recipe, portable PyTorch installation instructions, separate training/inference commands and the passing isolated execution trial. Copy that folder as a unit. The original champion binary is unavailable here; a full-budget retrain is documented separately from the small trial. The commands below are retained as historical instructions for a different, older deployment.

**Current status:** these commands document the preserved older 363-day deployment bundle. The research champion reported in the draft report has not yet been refitted on all available history. Before final handoff, align this guide with the audited final bundle. A Git clone currently excludes the fitted `submission/model.pkl`.

## Environment and installation

The preserved bundle documents Python 3.11, CPU inference and GPU training. Verify the exact dependency pins in a fresh environment on the chosen jury platform before release.

From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r submission/requirements-catboost.txt
```

Install before entering the offline evaluation environment. Provide the fitted artifact at `submission/model.pkl` alongside its matching manifest and code; no inference-time download is permitted.

## Inference

Run from the repository root, using one delivery date at a time:

```sh
python run.py submission/run.py --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

The root launcher resolves shared imports and file paths. Input history must contain only dates strictly before the evaluated date. All four zones and all 24 hourly prices are supplied together. Execute the same frozen pipeline separately for each jury date, with the corresponding history file.

| File | Required CSV columns |
| --- | --- |
| `test_hourly.csv` | `delivery_date,market_id,hour_index,hourly_price_eur_mwh` |
| `history_15min.csv` | `delivery_date,market_id,hour_index,quarter_index,price_eur_mwh` |
| `predictions.csv` | `delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh` |

Zones are `ZONE_0` through `ZONE_3`, hours are 1 through 24 and quarters are 1 through 4. A complete single day has 96 hourly input rows and 384 output rows. Each output hour must contain exactly four finite values with unique keys and a mean within 0.0001 EUR/MWh of the supplied hourly price. Predictions need not be monotonic across time; the ordered-quantile requirement concerns the optional uncertainty outputs.

## Retraining the preserved recipe

Use an authorized research server with a suitable GPU. This operation trains the older preserved recipe, not the current research champion:

```sh
python run.py submission/train_frozen.py --data train_prices_15min.csv --recipe submission/frozen_recipe.json --out results/retrained_frozen
python run.py submission/run.py --model results/retrained_frozen/model.pkl --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

The [frozen bundle README](../submission/README.md) records its original training setup and evaluation scope. The [champion refit plan](../docs/SUBMISSION_REFIT_PLAN.md) describes the remaining final-model work. Do not launch fitting to check an import or verify documentation.

## Clean-environment acceptance

Create a fresh environment and install only the documented requirements. Use a small fixture constructed from official historical dates, with every feature-history date earlier than the reconstructed date. Freeze the artifact and run inference with network access disabled. Inspect the CSV contract above and save the actual command, package versions, input identities, model hash, runtime outcome and output audit. A retrospective fixture checks execution and eligibility; it does not establish new held-out accuracy for an all-history refit.
