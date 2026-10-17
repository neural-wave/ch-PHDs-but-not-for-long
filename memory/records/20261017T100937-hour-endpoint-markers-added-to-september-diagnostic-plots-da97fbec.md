---
title: Hour endpoint markers added to September diagnostic plots
status: current
kind: visualization
recorded_at: '2026-10-17T10:09:37.155184+00:00'
sources:
- plot_error_hotspots.py
- results/error_hotspots_20261017/04_worst_examples.png
- results/error_hotspots_20261017/03_time_and_volatility.png
experiment_id: null
supersedes: []
---

User requested identification of first and last values within each hour. Updated price examples with open circle at Q1 (:00–:15) and open square at Q4 (:45–next hour), matching each observed/model curve color, plus alternating hourly shading and hourly x labels. Added matching markers to time-of-day MAE curves and explicit Q1–Q4 interval labels. Regenerated PNG/SVG and visually inspected both revised figures. Existing data assertions and score checks passed; model predictions unchanged.
