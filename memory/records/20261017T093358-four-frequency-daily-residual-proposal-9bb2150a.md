---
title: Four-frequency daily residual proposal
status: current
kind: experiment
recorded_at: '2026-10-17T09:33:58.175249+00:00'
sources:
- INTERPOLATION_HISTORY_PROTOCOL.md
experiment_id: null
supersedes: []
---

2026-10-17: User proposed four daily frequency weights. Design only; no training launched. Predict four coefficients per day/zone over fixed basis functions added to mean-corrected interpolation. Subtract each basis's within-hour mean to preserve known hourly averages. Free-phase Fourier modes generally need sine and cosine coefficients (eight real outputs for four frequencies); exactly four real weights requires fixed phases/bases. A regular 96-point day with 24 known hourly means leaves 72 residual degrees of freedom per zone, so four modes are an approximation rather than uniquely missing frequencies. Suggested next step: training-only basis selection, oracle held-out MAE reconstruction diagnostic, then causal coefficient prediction if capacity is adequate; compare rank-four learned residual modes and higher ranks. Oracle coefficients use targets solely for diagnostic capacity and cannot be used for deployment.
