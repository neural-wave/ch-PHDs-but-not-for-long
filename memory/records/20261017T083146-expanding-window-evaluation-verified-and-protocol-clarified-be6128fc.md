---
title: Expanding-window evaluation verified and protocol clarified
status: current
kind: analysis
recorded_at: '2026-10-17T08:31:46.120066+00:00'
sources:
- EXPANDING_PROTOCOL.md
- results/expanding_vast/learning_curves.csv
- results/expanding_vast/metrics.json
experiment_id: null
supersedes: []
---

On 2026-10-17, answered whether expanding chronological train/validation/test evaluation was tried. Verified saved CSV has 64 rows: eight expanding folds and eight fixed-September history lengths, four families each. Expanding starts Oct-Dec 2025 train / Jan 2026 validation / Feb test, ending Oct 2025-Jul 2026 train / Aug validation / Sep test. User's proposed one-month initial training would additionally test Dec 2025 and Jan 2026; those two folds are absent. Metrics JSON parses, smoke=false, lists 16 folds and reports boundary/normalization/causality/conservation/replay audits passed. No live remote job status was checked; startup's old active-status uncertainty should remain separate from available artifacts.

Method assessment: expanding-window walk-forward evaluation is appropriate for chronology. Fit learned preprocessing only on each fold's train; select candidates/epochs using that fold's validation and freeze before test; allow old test data into later training when historically available. Prior observed quarters must follow actual deployment availability. Aggregate held-out monthly MAE by target count and show per-month/per-zone outcomes and baseline-relative skill. Adjacent folds are dependent. Changing season and training size prevents interpreting expanding scores as a pure learning curve; existing fixed-September study addresses that separate question while adding older regimes. September already inspected, so all these are developmental scores, not a fresh blind holdout. No experiment launched or model changed.
