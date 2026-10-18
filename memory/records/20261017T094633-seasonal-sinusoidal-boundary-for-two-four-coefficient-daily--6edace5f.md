---
title: Seasonal sinusoidal boundary for two four-coefficient daily segments
status: current
kind: design
recorded_at: '2026-10-17T09:46:33.758922+00:00'
sources:
- FOUR_FREQUENCY_PROTOCOL.md
experiment_id: null
supersedes: []
---

2026-10-17 user asks to tune the switch between two four-coefficient daily segments with an annual sine wave. Proposed boundary in hours b(d)=b0+bs*sin(2*pi*d/T)+bc*cos(2*pi*d/T), equivalent to offset/amplitude/phase with three shared seasonal parameters; coefficients remain eight per day/zone. Bound boundary to a reasonable central-day range. Fit seasonal parameters using training days with temporally blocked/rolling validation and use August for final selection; one August alone weakly identifies a yearly curve. September remains developmental evaluation only. For an initial hard split round boundary to a whole hour, preserving hourly projection cleanly; for joint gradient fitting prefer sigmoid/overlap gate around the continuous seasonal boundary, then project the combined residual to zero within each hour. Smooth gate improves trainability and avoids a literal hard switch, but exact continuity is not guaranteed after hourly projection. Compare global-eight, fixed split four-plus-four, and seasonal split four-plus-four with identical predictor inputs/budget; assess prediction MAE as well as oracle capacity. This is a proposed extension, not implemented/launched in this turn. Annual calendar dependence is available without future-target leakage; its effectiveness has not been demonstrated.
