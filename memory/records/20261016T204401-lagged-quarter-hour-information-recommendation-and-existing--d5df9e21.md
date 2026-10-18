---
title: 'Lagged quarter-hour information: recommendation and existing features'
status: current
kind: analysis
recorded_at: '2026-10-16T20:44:01.377144+00:00'
sources:
- README.txt
- reconstruction.py
- MONTHLY_PROTOCOL.md
experiment_id: null
supersedes: []
---

2026-10-16: User asked whether to add lagged information given historical 15-minute granularity. Verified README permits strictly earlier delivery dates via official history_15min.csv; current-day quarters are excluded. Existing reconstruction.py already uses 1/2/7-day within-hour residual lags and 7/28-day trailing means. rich_features adds 14-day interpolation-error lags, trailing medians and similar-curve historical retrieval. Recommendation: retain historical within-hour shape information and test richer history with a matched chronological no-history / existing-history / expanded-history ablation selected on August. No incremental benefit from this recommendation was measured, and no experiment launched. Full raw-history sequence input is a possible new experiment, not current behavior. Preserve supplied hourly means and freeze inference history rules before evaluation.
