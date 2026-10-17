---
status: historical
---
# HACKATON project memory

Current objective (2026-10-16): save the three user-provided AET Neural Wave challenge files in this project and calculate quarter-hour reconstruction errors for hourly-mean repetition and linear interpolation between hourly observations.

Verified inputs: README.txt describes 363 days, four anonymized zones and 139,392 quarter-hour prices. The reference hourly price is the arithmetic mean of each hour's four quarters. Source files were copied from /Users/miguelnogales/Downloads/AET_CHALLENGE_PARTICIPANTS without overwriting existing files. No earlier project history was present.

Interpretation: evaluate deterministic reconstruction on the provided training data, not unseen test performance. Hourly means are available challenge inputs; quarter-hour targets are used only for scoring. Interpolate separately per delivery date and market to avoid needing future-day inputs. Document the placement of hourly observations and boundary handling explicitly. Do not interpolate the quarter-hour targets themselves, which would produce a meaningless zero error.

Next step: extract the challenge scoring definition, compute reproducible baselines, and save metrics and methodology. This is a lightweight descriptive calculation; no training, large evaluation campaign or remote job has been launched. Keep original data intact and distinguish source-document challenge rules from user instructions.

Completed 2026-10-16: evaluate_baselines.py validates the full grid and writes reproducible predictions, daily metrics and aggregate metrics under results/. MAE: flat hourly mean 7.5776104439 EUR/MWh; centre-aligned linear interpolation with constant daily boundaries 5.7382186702; per-hour mean-corrected interpolation 5.4741926007. Plain interpolation fails hourly-mean compliance; corrected interpolation passes to floating-point precision. See results/BASELINES.md and results/baseline_metrics.json. Next step is user-directed baseline/model development; no active jobs exist.
