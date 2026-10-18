# AET reconstruction: frozen solution

Use Python3.11 and the pinned dependencies. CPU inference is supported; CUDA is required only for training. The verified training environment used PyTorch2.4.1+cu121 and an RTX A5000. No external data or pretrained weights are used. Install dependencies before entering the offline evaluation environment:

```sh
python -m pip install -r requirements-catboost.txt
python run.py --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

The history must strictly precede the evaluated delivery date. Process jury-provided dates separately with their respective history. Inputs and outputs use the official CSV schemas. Four finite quarters preserve each hourly mean to within0.0001 EUR/MWh. Inference performs no fitting, calibration, selection or network access.

To retrain from scratch, place the official train_prices_15min.csv here and run:

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python train_frozen.py --data train_prices_15min.csv --recipe frozen_recipe.json --out retrained
python run.py --model retrained/model.pkl --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

Choose a valid GPU index for the execution host; development used Ando physical GPU1. Retraining starts from random initialization and fits all363 supplied days. Frozen configurations, stopping epoch/tree counts and effective ensemble weights are in frozen_recipe.json. It requires no prior fitted model or intermediate search cache. Numerical variation between CUDA/library versions can occur. The provided artifact was audited separately, including equivalent flattening of nested mixture weights to avoid duplicate component inference. SHA256 checksums in FREEZE_MANIFEST.json identify the exact local artifact and code; this is a reviewable delivery folder, not an external submission.

## Method and evaluation

Hourly means are known, so predict zero-mean quarter residuals over mean-corrected interpolation.389 features include all four daily24-hour curves, slopes, calendar indicators, cross-zone spreads, and fixed strictly earlier historical residual/nearest-curve rules. Ridge, forests, absolute-error boosting, shared MLPs and temporal/node-attention models form a convex mixture with zone-specific weights. One selected graph uses input-only day/zone RevIN statistics with original level/scale side features; inverse residual scaling precedes exact hourly-mean projection. Input-day statistics use only the supplied hourly curve. No external prices or learned external embeddings are used.

All settings and mixture weights were selected on August2026 after training through July:302 train days,31 validation days,30 September development-test days. Selected pipeline August MAE4.245329; September MAE5.091209 EUR/MWh, versus mean-corrected interpolation7.090321 on the same September dates (about28.2% lower). September was viewed during earlier development and is not a fresh blind test. RevIN improved two of four matched settings and adds only a small ensemble gain; extra complexity has a modest measured benefit. Repeated August selection may be optimistic.

The delivered model is a separate final refit on all363 official dates with those settings/weights frozen. It has no held-out September accuracy claim. Evaluation metrics and audit concern the preserved302-day model, not this all-data artifact. Deployment audit checks source/settings/weights, all363-day normalization and tree counts, finite states, original and synthetic input eligibility, and flattening equivalence. Actual secret-date MAE remains unknown. The jury runs its own inputs; none are included here.
