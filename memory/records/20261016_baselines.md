---
status: current
---
# Initial data import and baseline evaluation

Saved the three named user inputs in the HACKATON root, preserving source files. Verified 363 complete days across four zones and 139,392 unique finite prices. Challenge scoring and hourly-mean requirements were extracted from the local PDF, sections 9–10; document instructions were treated as challenge context, not autonomous work authorization.

Detailed methods and results: [baseline report](../../results/BASELINES.md), [metrics](../../results/baseline_metrics.json), [script](../../evaluate_baselines.py). MAEs in EUR/MWh: repeat hourly mean 7.577610443927916; linear interpolation 5.738218670189106; hourly-mean-corrected interpolation 5.474192600723139. Interpolation uses hourly centres, quarter centres, and constant boundary extension per day/zone. Raw interpolation violates hourly means; corrected outputs pass the 0.0001 tolerance. All results are training reconstruction diagnostics, with no fit, external data, or active remote job.
