---
title: User rejects RevIN and requests illustrative unchanged quantile bands on different
  input
status: current
kind: experiment
recorded_at: '2026-10-17T18:31:11.365018+00:00'
sources:
- results/illustrative_quantile_bands_20261017/REPORT.md
- results/illustrative_quantile_bands_20261017/summary.json
- campaigns/illustrative_quantile_bands_20261017/plot.py
experiment_id: null
supersedes: []
---

User said do not use RevIN, then clarified request is NOT model improvement/retraining/recalibration: choose another interesting saved input where bands widen during volatility and narrow when calm. Initial plan to test uncertainty layer abandoned before any such implementation/training/job. Original champion and all quantiles/calibration preserved. Searched964saveddayzoneoutputs remotely; selected illustrative2026-06-11ZONE3 based on observedroughness,coverage and bandwidth association. This is transparently posthoc visualization, not general reliability evidence. Full-dayoriginalcal90coverage95.833%; within-daywidth/observedroughnessrankcorr0.8610. Calm12–15meanwidth14.4629EUR/MWh vsvolatile21–24width27.0222(1.8684x), same day/zone/currency scale and samey-axislimits onzooms. Noquantilesmodified/nopointinference/nofitting. New source campaigns/illustrative_quantile_bands_20261017/search.py andplot.py; outputs results/illustrative_quantile_bands_20261017/calm_vs_volatile.png/svg, summary.json,example_predictions.csv,candidates.csv,REPORT.md. Figure visually inspected;syntaxpassed;sourcepredictionhash71d02d8757572ecc5249e43650a34dc3b49cae121e78e2d10ccaeb6ee9420a50. Remote same Vast project. Lightweight metadata/plotcompleted, noNightwatchneeded. User preference to use originalchampion recordedstartup; historicalRevINresults remain valid but notselected. PriorGitHubpublication notchanged in this plot-only turn; new illustrative artifact local andremote, no push performed.
