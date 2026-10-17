# Raw-hourly zonal CNN

This repository contains the per-zone CNN trained with the 24 raw hourly prices as its only input channel. The input prices are not normalized and the model uses no derived profile features or calendar embeddings. The saved checkpoints are in `experiments/raw_hourly_only_zonal_cnn/models/`; `run_info.json` and `validation_metrics.csv` record the selected configurations and August–September results.

The models use two Conv1d/GELU/Dropout blocks followed by a 1×1 Conv1d head that predicts four quarter-hour residuals per hour. The predicted residuals are scaled using a train-only residual scale and centered so each hour's four quarter-hour predictions average to the supplied hourly price. The training objective is Q1/Q4-weighted MAE with quarter weights 1.5, 1, 1, 1.5 and month-difficulty weights of strength 0.5.

The training set is `train_prices_15min.csv`; supporting zone-identification data is in `zone_identification/`. The October validation CSVs cover October 1–16 only. The August–September validation months are included in the training CSV and are held out by the scripts.

## Reproduce the experiment

Clone the repository and enter its root directory:

```sh
git clone git@github.com:neural-wave/ch-PHDs-but-not-for-long.git
cd ch-PHDs-but-not-for-long
```

Use Python 3.12, install the pinned dependencies, and run the month-weighting step before training the raw-hourly models:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/refit_month_difficulty_no_month_embedding.py
python scripts/train_raw_hourly_only_zonal_cnn.py
```

The first script scores each held-out month with an 11-month CNN fold, computes the per-zone difficulty scores and October–July training weights, and writes them to `experiments/month_difficulty_no_month_embedding_aug_sep/`. It reuses the committed LOMO scores in `monthly_oof_difficulty.csv` by default so the same month weights are reproduced directly. To rerun all 96 scoring folds, remove that file before running the script; it will regenerate the scores and weights from `train_prices_15min.csv`. The second script reads those weights and trains the four raw-hourly zonal CNNs, writing checkpoints, metrics, and validation predictions to `experiments/raw_hourly_only_zonal_cnn/`.

Both scripts use Apple MPS automatically when available and otherwise run on CPU. Add `--require-mps` to either command to require MPS. The scripts, data, selected CNN configurations, old comparison weights, saved LOMO scores, and resulting model checkpoints are included in the repository. The October validation data is separate and is not used by these two training steps.
