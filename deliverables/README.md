# PhDs but not for long — reproducible ensemble

Standalone training and offline inference for the final six-network ensemble:
two compact MLPs, two temporal CNNs and two CNNs over paired historical
quarter/hour grids. Each network has weight 1/6. Predictions correct
mean-conserving linear interpolation; the point prediction is the ensemble q50.

Team: Daniel S. Vega, Valentina Moretti, Riccardo Ghetti and Miguel Nogales.

This folder can be copied out of the research repository. It needs no research
server, Git checkout, cached features or files from its parent directory.
Original research scripts and the frozen submission were left in place.

**Checkpoint status:** the original frozen champion weights were not available
for this package. `model_checkpoints/trial/` contains the completed small
execution trial described below. A trial checkpoint is explicitly labeled and requires
`--allow-trial` at inference. It is not the fitted final champion and establishes
no accuracy result. The full training command regenerates the final **recipe**;
it does not promise the byte-identical original checkpoint across hardware.

## 1. Install Python and choose PyTorch for your machine

Use **Python 3.12** and a fresh environment. From this folder on Linux/macOS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Install ONE PyTorch option below, then install the common requirements.**
`requirements.txt` contains NumPy and pandas only: it will not overwrite your
chosen PyTorch build. The optional `requirements-torch-reference.txt` pins the
reference release, `torch==2.14.1`, without a macOS wheel URL, architecture tag,
CUDA suffix or `+cpu` suffix. Pip selects a compatible wheel for the running
Python and operating system.

### Linux or Windows, CPU only

CPU inference is the portable reference path and does not require a GPU:

```sh
python -m pip install -r requirements-torch-reference.txt --index-url https://download.pytorch.org/whl/cpu
```

### macOS

Install from PyPI; do not use a CUDA index on a Mac:

```sh
python -m pip install -r requirements-torch-reference.txt
```

The reference release provides a macOS 14+ Apple Silicon wheel. For a machine
that cannot install this release, choose a compatible PyTorch 2.x release/build
using the official instructions below, and run the execution trial to verify
that environment. Availability of wheels depends on the OS, architecture and
Python version. Inference here runs on CPU, including on Apple Silicon.

### NVIDIA GPU on Linux or Windows

Open the [official PyTorch installation selector](https://pytorch.org/get-started/locally/),
select your OS, **Pip**, **Python**, and a CUDA build compatible with your GPU
driver. Run its `torch` installation command in the activated environment.
Only `torch` is needed; this project does not use `torchvision` or `torchaudio`.
For the reference release, consult the
[official version-specific instructions](https://pytorch.org/get-started/previous-versions/)
and select an available CUDA wheel for 2.14.1. A CPU-only PyTorch installation
cannot run `--device cuda`; it can run all commands with `--device cpu`.

You may install another compatible PyTorch 2.x build (2.4 or newer) for your machine
instead of the optional reference pin. Re-run the trial below when changing
PyTorch or numerical dependencies. This is compatibility checking, not a claim
of bitwise identical GPU/CPU training.

### Common requirements and verification

After installing your chosen PyTorch build:

```sh
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch,numpy,pandas; print('torch',torch.__version__,'CUDA build',torch.version.cuda,'CUDA available',torch.cuda.is_available()); print('numpy',numpy.__version__,'pandas',pandas.__version__); print(torch.ones(2).numpy())"
python -m pip freeze > installed-environment.txt
```

The original packaged reference environment was Python 3.12.15, NumPy 2.5.4,
pandas 3.0.6 and Linux PyTorch 2.14.1+cpu. It is recorded as historical evidence
in `provenance/original_environment.json`. Use the installation choice for your
machine rather than installing a frozen environment exported from macOS.

For a fully offline machine, prepare a wheel directory **on a machine with the
same OS, architecture, Python version and selected PyTorch build**. For example,
for a Linux/Windows CPU environment:

```sh
python -m pip download --dest wheelhouse -r requirements-torch-reference.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip download --dest wheelhouse -r requirements.txt
```

Transfer this folder and `wheelhouse/` to the offline machine, activate its fresh
environment, then install without network access:

```sh
python -m pip install --no-index --find-links wheelhouse -r requirements-torch-reference.txt -r requirements.txt
```

For macOS, omit the CPU `--index-url` while downloading. For CUDA, use the chosen
CUDA index/release. Wheels prepared on macOS cannot be reused on Linux/Windows.
After installation, neither training nor inference makes network requests.

## 2. Run the isolated execution trial

```sh
python src/verify_pipeline.py --output model_checkpoints/jury_trial
```

This copies the standalone package into a temporary directory outside the
repository, fits **all six full-size architectures** for one epoch each on the
first four official dates, calibrates on the next two dates, and predicts the
seventh date from its hourly means and strictly earlier observed quarters.
Training and prediction run with socket networking disabled. It checks the CSV
contract, conserved means, quantile ordering, nested calibrated intervals,
checkpoint replay, missing history and rejection of invalid inputs. No
evaluation-period prices are used and the execution trial does not evaluate
accuracy. Use a fresh output path, such as `model_checkpoints/trial-rerun`, to
repeat it without overwriting a previous trial. The author's completed check is
already recorded in `model_checkpoints/trial/`; the command above uses a new
directory so it also works when that example checkpoint is included.

The output contains `model.pt`, training/environment records, `verification.json`,
logs and `fixture/` input/output CSVs. To run its prediction entry point yourself:

```sh
python src/predict_prices.py --model model_checkpoints/jury_trial/model.pt --allow-trial --hourly model_checkpoints/jury_trial/fixture/test_hourly.csv --history model_checkpoints/jury_trial/fixture/history_15min.csv --output model_checkpoints/jury_trial/replayed_predictions.csv --quantiles-output model_checkpoints/jury_trial/replayed_quantiles.csv
```

## 3. Retrain the complete final recipe

This command is provided for jury reproduction; it is separate from the tiny
execution trial. It performs all fixed epoch budgets and can take substantially
longer. Use an otherwise empty output directory:

```sh
python src/train_ensemble.py --mode final --device cpu --data data/train_prices_15min.csv --output model_checkpoints/final
```

With a working CUDA installation, replace `--device cpu` with `--device cuda`.
The same script runs on both devices. There is no dependency download, remote
connection, model search or calibration-based early stopping in this command.

The source data must match the official SHA-256:
`f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c`.
Training uses 348 eligible dates, 2025-10-01 through 2026-09-15. The 15 dates
2026-09-16 through 2026-09-30 are held out for interval calibration. The two
missing DST dates remain absent. All features are rebuilt directly from the
CSV with strictly earlier quarter-hour history; hourly inputs are averages of
that day's official quarters. Fixed scaling and feature rules are unchanged.

| Component | Seed | Epochs | Learning rate | Ensemble weight |
| --- | ---: | ---: | ---: | ---: |
| Compact MLP | 17 | 112 | 0.003 | 1/6 |
| Compact MLP | 43 | 134 | 0.003 | 1/6 |
| Temporal CNN | 43 | 33 | 0.003 | 1/6 |
| Temporal CNN | 17 | 54 | 0.003 | 1/6 |
| Historical-grid 2D CNN | 17 | 199 | 0.001 | 1/6 |
| Historical-grid 2D CNN | 43 | 176 | 0.001 | 1/6 |

The frozen recipe is `config/final_recipe.json`. Training uses batch size 32,
AdamW with weight decay 0.01, gradient clipping at 5, and median MAE plus eleven
quantile pinball losses. Per-month weights use fitting dates only. NumPy and
PyTorch seeds are reset for each component; TF32 is disabled and PyTorch uses
one CPU thread. Component settings and epoch budgets were selected during
development; this command reproduces the final fit rather than rerunning the
entire research search.

`model.pt` contains all six state dictionaries, mixture weights, the recipe and
the held-out interval corrections. Separate JSON records and checksums describe
the run. A freshly loaded CPU ensemble must reproduce calibration predictions
within 0.0001 EUR/MWh before training is reported complete. Hardware and library
versions can affect freshly fitted weights; preserve the generated environment
record alongside the checkpoint.

## 4. Predict the jury inputs, offline

After the full retrain, or after obtaining the matching original champion
checkpoint and putting it at the explicit model path:

```sh
python src/predict_prices.py --model model_checkpoints/final/model.pt --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
```

An optional `--quantiles-output quantiles.csv` writes the eleven marginal
quantiles and calibrated 50/80/90/95% intervals in a separate file. Submit the
five-column `predictions.csv` for point scoring. Inference only reads the fitted
checkpoint and the two input CSVs; it never fits, recalibrates or downloads.

Input headers:

```text
test_hourly.csv: delivery_date,market_id,hour_index,hourly_price_eur_mwh
history_15min.csv: delivery_date,market_id,hour_index,quarter_index,price_eur_mwh
```

Each requested date needs all 4 zones × 24 hours, with hours numbered 1–24.
History dates must be strictly earlier than the first requested date. A history
date needs all 4 × 24 × 4 observations; missing calendar dates are allowed and
are handled by the frozen availability masks. An empty history file with the
correct header is supported. For successive jury dates with updated observed
history, call the command once per date using the supplied history for that
date. The code never treats predicted quarters as observations.

The point output has exactly these columns:

```text
delivery_date,market_id,hour_index,quarter_index,predicted_price_eur_mwh
```

There are 384 unique finite rows per requested date, four quarters per hour.
Their mean equals the supplied hourly price within 0.0001 EUR/MWh. Rows are
ordered by date, zone, hour and quarter. Only q50 is constrained to the hourly
mean; the other marginal quantiles describe uncertainty.

The independent CSV validator can also be run on the jury's files:

```sh
python src/check_predictions.py --hourly test_hourly.csv --predictions predictions.csv
```

## 5. Package the files for transfer

To package the completed execution trial, including its ignored-by-Git weights:

```sh
python src/package_pipeline.py --kind trial --output ../jury_pipeline_trial.zip
```

After the full retrain or receipt of the original champion, package the final
artifact instead:

```sh
python src/package_pipeline.py --kind final --output ../jury_pipeline_final.zip
```

The script includes this folder's code, data, documentation and the selected
checkpoint directory, excludes virtual environments/caches, verifies every
archived file against its hash, and writes a `.sha256` sidecar. It refuses to
package a trial as a final model or overwrite an existing ZIP. A different
checkpoint location can be selected with `--checkpoint-dir`, relative to this
folder and inside `model_checkpoints/`. Download/install dependencies separately
as described in section 1; the archive does not bundle platform-specific wheels.

## Checks completed in this checkout

The fresh Python 3.12.15 environment with PyTorch 2.14.1, NumPy 2.5.4 and pandas
3.0.6 passed `pip check` and all 14 isolated execution checks on Apple Silicon
macOS, using CPU inference. The one-day and two-day outputs contain 384 and 768
valid rows; the maximum CSV hourly-mean error was approximately
`2.6e-11 EUR/MWh`. Checkpoint replay was exact. Details and logs are in
`model_checkpoints/trial/verification.json` and `invariants.json`.

All 22 retained model/feature definitions match the original source AST after
local import renaming. Using the same trial state dictionaries, the original
frozen runtime and the renamed runtime gave identical features and all eleven
ensemble quantiles with complete, empty and partially missing prior history.
See `verification/source_equivalence.json`.

The reference PyTorch installation was also resolved for Linux x86-64 CPU and
Windows x86-64 CPU: pip selected each platform's own `2.14.1+cpu` wheel. These
are dependency-resolution checks, not execution tests on those operating
systems. See `verification/platform_installation.json`. CUDA execution and the
full 348/15 final retrain were not performed during this local trial.

## Files and provenance

| Folder/file | Purpose |
| --- | --- |
| `data/train_prices_15min.csv` | Unchanged copy of the official training input |
| `config/final_recipe.json` | Unchanged frozen recipe, including seeds and budgets |
| `src/train_ensemble.py` | Training from official CSV; explicit final/trial modes |
| `src/predict_prices.py` | Offline CPU inference entry point |
| `src/verify_pipeline.py` | Isolated six-component end-to-end trial |
| `src/check_predictions.py`, `src/package_pipeline.py` | Independent CSV validation and checked transfer ZIP |
| `src/quantile_models.py` | Copied quantile architectures and interval calibration |
| `src/feature_pipeline.py` | Copied feature assembly and month weights |
| `src/price_data.py`, `compact_features.py`, `historical_errors.py` | Copied baseline and causal features |
| `src/history_grid.py`, `analog_cnn.py` | Copied quarter/hour grid and 2D CNN building blocks |
| `model_checkpoints/` | Final and trial artifacts kept in separate directories |
| `docs/challenge_rules.pdf`, `docs/RULES_AND_METHOD.md` | Organizer rules and method/validation notes |
| `provenance/source_map.json` | Original paths, source hashes, retained symbols and copy names |
| `provenance/code_inventory.json` | Static inventory of all 268 Python files in the source checkout |
| `provenance/original_*` | Historical frozen-bundle evidence; not audits of this trial |

Sources were traced to commit
`36c172c604d7f1e9c98b62ba6199f8ca2ee3e1e7` of the team's authorized research
branch. Only the final dependency closure is copied here; unrelated search,
plotting, remote launch and alternative-model runners are omitted. Definitions
for feature/model calculations retain their original bodies, with renamed local
imports. The new trainer replaces a missing research cache with direct feature
generation. The original deployment identifier is
`champion_348fit_15cal_20261017`; the original model SHA-256 is
`6ee16ba8942f3ad569ceaf1f5f578608801a0d668ea6b4d39398709d2d4e5a5d`.

When transferring this package, include the fitted checkpoint you intend to
deliver: `*.pt` files are excluded from the parent research repository's Git
tracking. A source-only clone therefore needs the full retrain before final
inference. The report and recorded video remain separate submission materials.
