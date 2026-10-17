---
title: September error hotspots for frozen joint blend and simple family winners
status: current
kind: analysis
recorded_at: '2026-10-17T10:00:28.210625+00:00'
sources:
- plot_error_hotspots.py
- results/error_hotspots_20261017/summary.json
- results/monthly_joint_blend_gpu1/test_predictions.npz
- results/simple_5090/selection.json
experiment_id: null
supersedes: []
---

Generated and visually inspected four PNG/SVG plots from frozen September predictions on 2026-10-17. No model training or selection performed. Compared August-selected joint blend (MAE5.091209), family-selected linear (5.209794), CNN (5.222494), interpolation (7.090321); checked official dates/targets, reported scores, finite predictions and hourly-mean preservation. September is previously inspected development data, not blind. Joint blend zone MAEs 3.8731,6.7331,5.3934,4.3653; first/last quarter MAE6.1593/6.1355 versus middle4.0632/4.0069. Worst day-zone September5 ZONE_1 MAE13.6688, then September28 10.6201 and September4 10.3388. Greatest point error September22 ZONE_0 hour_index20 quarter4: observed533.26, predicted658.5095, absolute125.2495. Worst5% points account for25.5812% absolute error. Observed within-hour range40+ has MAE8.6980 versus1.5973 at0–5; this grouping uses targets for diagnosis only. Top hourly-average MAE at hour_index18 (8.5859), dataset plotted as hour start17. Four example panels include three hardest day-zones and largest point-error day. Output results/error_hotspots_20261017/{01_zone_and_day,02_day_hour_heatmap,03_time_and_volatility,04_worst_examples}.{png,svg}, summary.json. Lightweight local plotting only; no remote campaign needed. Reproduce with temporary matplotlib environment used in this turn. Startup summary archived before shortening in memory/records/20261017_startup_before_error_hotspots.md.
