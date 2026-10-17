---
title: Monthly baseline plot split by anonymized market
status: current
kind: experiment
recorded_at: '2026-10-16T21:42:41.052041+00:00'
sources:
- plot_monthly_baselines_by_zone.py
- results/monthly_baseline_mae_by_zone.csv
experiment_id: null
supersedes: []
---

User requested country split of the original three baseline monthly plot. Official package identifies markets only as ZONE_0–ZONE_3, so preserved anonymized labels without guessing country identities. Generated shared-scale 2x2 monthly plot and CSV, validating 48 month/zone combinations and 363 days per zone; reaggregated zone MAEs match original monthly chart within 1e-10. Mean-corrected interpolation wins all 48 combinations. Outputs results/monthly_baseline_mae_by_zone.png, .svg and .csv; reproduction script plot_monthly_baselines_by_zone.py. No model changes or remote job actions.
