# PhDs but not for long — champion inference

## Model and verified environment

Use deployment recipe **`champion_348fit_15cal_20261017`**: six freshly fitted networks, equal MLP/temporal-CNN/analog-CNN family weights, two seeds per family. Fit: 348 official dates through September 15, 2026; calibration: September 16–30. Inference uses CPU, performs no training or recalibration, and makes no network requests.

On 17 October 2026 the research-server receipts reported successful clean dependency installation, isolated CPU checkpoint replay and a 384-row CSV fixture audit. Verified environment: **Linux, Python 3.12.15, PyTorch 2.14.1+cpu, NumPy 2.5.4, pandas 3.0.6**. Other operating systems have not been verified. Maximum fixture hourly-mean error: **2.504e-11 EUR/MWh**. This checks execution, not future predictive accuracy or coverage.

The matching archive was recovered locally and its SHA256 checked. Download/transfer the fitted archive separately from the repository: model binaries and archives are excluded from Git. The final public artifact delivery link is still pending. The older `submission/model.pkl` is a different model and is not used by these instructions.

## 1. Unpack the complete bundle

Place `champion_submission.tar.gz` in your working directory. Its SHA256 must be:

```text
63d96671184d92f81fec6b71d89d049cd215795fe80759e7fe5a75c6c1f5abba
```

```sh
sha256sum champion_submission.tar.gz
tar -xzf champion_submission.tar.gz
cd champion_submission
```

The folder must contain `run.py`, `model.pt`, `sources/`, `requirements.txt`, `recipe.json`, `calibration.json`, `environment.json`, `audit.json`, `SPECIFICATION_CARD.md` and `MANIFEST.json`. Keep these files together. Use the trusted team-supplied artifact, whose checkpoint embeds its calibration.

Verify every packaged file before creating the environment:

```sh
python3.12 - <<'PY'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('MANIFEST.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Bundle checksums verified')
PY
```

## 2. Install once

On the verified Linux/Python 3.12 platform:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

Install dependencies while internet is available, before offline evaluation. No GPU, CUDA, CatBoost, repository checkout or training data download is required for this standalone bundle. Dependency installation is separate from inference.

## 3. Run inference

From inside `champion_submission/`, with the environment active:

```sh
python run.py \
  --hourly /absolute/path/test_hourly.csv \
  --history /absolute/path/history_15min.csv \
  --output /absolute/path/predictions.csv
```

Create output parent folders beforehand. The checkpoint defaults to the `model.pt` beside `run.py`; `--model /absolute/path/model.pt` can explicitly select that matching artifact. This command is the standalone bundle's entry point, not the repository's root launcher.

### Input CSV contract

| File | Required columns |
| --- | --- |
| Hourly input | `delivery_date,market_id,hour_index,hourly_price_eur_mwh` |
| Observed history | `delivery_date,market_id,hour_index,quarter_index,price_eur_mwh` |

- Dates: ISO `YYYY-MM-DD`. Zones: `ZONE_0`, `ZONE_1`, `ZONE_2`, `ZONE_3`.
- Hours: integers **1–24**. Quarters: integers **1–4**. Prices: finite numeric EUR/MWh.
- Every requested date needs **96 unique hourly rows**: all four zones × all 24 hours. Row order does not matter.
- Each supplied history date must be complete: **384 unique observed quarters**. Missing historical dates are supported; incomplete days are not the expected format.
- **Every history date must precede the earliest requested date.** Do not include observed quarters from the day being reconstructed.
- Supply the available official historical observations. The known full-day hourly curves are allowed; current-day quarter observations are not.

For several future dates, one call can include all their hourly curves, using history strictly before the first date. Predictions are never inserted into observed history. To incorporate newly observed earlier days, run each date separately with its updated, strictly prior history file.

### Point output

```text
delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh
```

The output contains **384 rows per requested date**, one finite price per zone/hour/quarter. The four predictions per hour average to the supplied hourly price within **0.0001 EUR/MWh**. The point decision is the ensemble median (q50). Predictions may rise or fall across quarters; temporal monotonicity is not required.

## 4. Optional uncertainty output

```sh
python run.py \
  --hourly /absolute/path/test_hourly.csv \
  --history /absolute/path/history_15min.csv \
  --output /absolute/path/predictions.csv \
  --quantiles-output /absolute/path/uncertainty.csv
```

`predictions.csv` retains exactly the five point-output columns. `uncertainty.csv` additionally includes:

- Eleven ordered raw quantile columns: `q0.025,q0.05,q0.1,q0.2,q0.25,q0.5,q0.75,q0.8,q0.9,q0.95,q0.975`.
- Calibrated interval endpoints: `lower_50,upper_50,lower_80,upper_80,lower_90,upper_90,lower_95,upper_95`.

Only q50 has the hourly-mean conservation guarantee. Other quantiles and marginal interval endpoints need not preserve the hourly mean. Historical empirical coverage does not guarantee coverage on future dates.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing `model.pt` or `sources` import failure | Unpack the whole fitted archive and keep its layout intact; a Git clone alone is insufficient. |
| Dependency wheel cannot be found | Use the verified Linux/Python 3.12 environment and the unmodified CPU requirements. |
| `Duplicate hourly keys` | Provide one row per date/zone/hour. |
| Nonfinite or incomplete hourly input | Supply all 96 finite hourly values for each date and the exact zone/hour labels. |
| `History must precede the first input date` | Remove current/future observed dates from the history file. |
| Output path error | Create the output folder and check write permissions. |

Preserve the original frozen `submission/` bundle and its manifest. Final delivery should include this champion archive, its checksum, this guide and the report/video locations. Research performance provenance is in [BEST_MODEL.md](../BEST_MODEL.md); the deployment specification is in [the campaign card](../campaigns/submission_champion_20261017/SPECIFICATION_CARD.md).
