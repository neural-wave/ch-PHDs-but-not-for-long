# Completion audit for the requested optimization

Interpret the user's correction as **minimize** error relative to interpolation.
The requested expanding-training regime remains the governing scope. The
scientific campaign, local recovery backup and final comparison are complete
and verified. Repaired backup Nightwatch a6689a6ede6fcf6102ea8476 completed
with exit 0 on 17 October 2026 at 18:06:39 Zurich.

| Requirement | Owning evidence and verification | Status |
|---|---|---|
| Optimize the new primary metric | Full 384-fit frozen campaign; verified_selector_comparison.json independently recomputes selector ratio 0.721285 → 0.708892 from matched quarter predictions. | Verified |
| Increasingly wide training sets | selector_comparison_by_training_size.csv: eight folds, October 2025 start throughout, 3–10 months and 91–302 fitting days; all 241 test days retained. Source experiment.py constructs expanding masks. | Verified |
| Official data and permitted causal inputs | Source/official CSV hashes rechecked locally; prepare repeats future-target perturbation and truncation probes. Full audit matches saved targets, dates and baseline to official source. Current hourly curves are permitted inputs. | Verified |
| Fit from scratch; exclude selection/calibration/test targets from gradient fitting | trainq.py initializes using y[:nt] and samples optimizer IDs from randperm(nt). Only training plus selection arrays enter early stopping. Source hashes match the audited execution manifest. | Verified |
| Select before the next test month | train.py uses preceding selection truth only for candidate/seed-average/blend choices, saves selection before evaluate; audit_selection.py reconstructs exact menus and argmins. No test-label recipe selection. | Verified |
| Training-only preprocessing and objective weights | Fixed engineered units/clip15; fitting-only additive knots and training-month ratio weights are reconstructed by audit.py. Frozen standard controls independently replay their original training-data transformations. | Verified |
| Hourly means conserved | Every candidate's q50 checked, 281 selected new/control checkpoint replays; maximum hourly-mean error 1.14e-13 EUR/MWh, below 0.0001. | Verified |
| Default eleven ordered quantiles and calibrated 50/80/90/95% intervals | Audit checks all levels, order, nonzero widths, physical replay, separate days16–end CQR ranks, widening, nesting, coverage and confidence labels. selector_interval_comparison.csv independently recomputes interval statistics. | Verified |
| Equal-month score, raw MAE, per-month/zone scores and alternate denominator | audit.py verifies monthly and primary aggregate errors; monthly_zone_scores.csv retains identical-subset zone errors; aggregate_baselines.py verifies aggregation before forming corrected/raw interpolation ratios. | Verified |
| Distinguish pre-test selector and aggregate grid winner | Primary 0.708892 belongs to the pre-test policy; 0.707968 fixed_blend_ratio is reported separately as the descriptive aggregate winner. INTERPRETATION.md preserves this distinction. | Verified |
| Preserve uncertainty and developmental caveats | INTERPRETATION.md reports coverage/width/error groups and temporal dependence; training-size/calendar confounding and joint preprocessing/seed/blend changes are explicit. No blind-benchmark, significance or global-optimum claim. | Verified |
| Execute on the authorized research GPU with actual success/failure tracking | remote_terminal_state.json: Nightwatch0c174584a3b9236d9f21ffd4 completed exit0 at17:36:04Zurich. Full audit plus completion receipt inspected. No training was run on the Mac. | Verified |
| Save reproducible models, sources, raw quantiles and interval predictions | Remote recovery archive 2,428,656,306 bytes; completion.json SHA2566263d9223b7bcb7c9b4855ced55a74e4844ea1463d175866a5072d08236701d0; archive includes new models and frozen-standard replay dependencies. | Verified remotely |
| Local recovery transfer and checksum | backup_verification.json confirms 2,428,656,306 bytes and SHA2566263d9223b7bcb7c9b4855ced55a74e4844ea1463d175866a5072d08236701d0. Repaired job a6689a6ede6fcf6102ea8476 completed exit0; metadata hashes and atomic rename independently checked. | Verified |
| Reviewable final report and plot | INTERPRETATION.md, REPORT.md, recomputed CSV/JSON comparisons, corrected PNG/SVG and plot_verification.json saved. Plot visually checked after correcting overlapping labels. | Verified |
| Project records and startup summary | Final result and checksum records retained; earlier pending-status records marked historical. Prior startup preserved verbatim, current short summary updated and project RAG refreshed. | Verified |

The optimized score is a measured developmental result from this finite search,
with improvement in seven of eight months. It is not proof of mathematical
global optimality. Production/deployment replacement was outside the requested
comparison and was not performed.
