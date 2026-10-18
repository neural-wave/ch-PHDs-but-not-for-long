---
title: Parallel 2D CNN completed with audited metrics and inspected learned filters
status: current
kind: result
recorded_at: '2026-10-17T11:27:41.622707+00:00'
sources:
- results/parallel_2d_zone_cnn_20261017/REPORT.md
- results/parallel_2d_zone_cnn_20261017/audit.json
- results/parallel_2d_zone_cnn_20261017/metrics.json
- results/parallel_2d_zone_cnn_20261017/trials/3/kernels_channel_0.png
- results/parallel_2d_zone_cnn_20261017/trials/3/activations_august01.png
experiment_id: parallel_2d_zone_cnn_20261017
supersedes: []
---

Verified2026-10-17: user explicitly confirmed illustrated parallel Q/H rows match intent. Nightwatch1b9cf34ee35a62b780645756 completed exit0, ended1792234022.682 (12:47Zurich), wrapper5024 absent; full6fits, independent audit and filterplotexport completed. August-selected no-history seed71 trial2:940params, August4.8938486274, September5.9037670836. History7days seed17 trial3:1388params, August4.6050121533, September5.4311269740. All4zones joint, hourly maxerror5.684e-14, replay0; sourcehash/causality/train-only normalization/selection/metricreconstruction checks passed. History2D September1.965% lower than minimal1D history5.539996, though priorjointblend5.091209 remains better. September developmental. No deployment. Copied all six checkpoints, selected predictions, reports/metrics/audit/manifests, raw learned+fusionweights and plots to local results/parallel_2d_zone_cnn_20261017 (feature/data NPZ excluded from backup; official input exists locally). Remote root@93.91.156.85:55340,/workspace/HACKATON; log/root/.local/share/nightwatch-hackaton/1b9cf34ee35a62b780645756.log. Visually inspected trial3 zone0 kernelheatmaps and August1activations: readable labels/layout. Each of8filters has2x8 Q/H normalized-input weights perzone; interpretations must account for normalization and downstream nonlinear fusion. Plotlabels explicitly avoid causal claims. Querydayquarterrow always masked; itshourlyrow available. Daymixing in tinyfusion,2Dkernelspanspairedrepresentations. No numerical attribution or perdayablation beyondmatchedhistory comparison yet. Startup preserved memory/records/20261017_startup_before_parallel_2d_results.md.
