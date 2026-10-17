# HACKATON project memory

Objective (2026-10-16): minimize quarter-hour MAE with official inputs only; no pretrained/external/October prices. Four zones jointly; hourly means within0.0001 EUR/MWh. Train Oct–July302 days, August31 validation, September30 development test(previously viewed). Data363 days/139,392 quarters.

Prior monthly winner monthly_joint_blend_gpu1: August4.245329, September5.091209;11,520-row serialized/CSV/causal audit passed. Evidence results/monthly_joint_blend_gpu1/audit.json, results/monthly_best.json; results/experiments_summary.html. RevIN wins2/4 pairs, no broad superiority. Original context: memory/records/20261016_before_hpo_lags_startup.md.

All363 frozen deployment refit on AndoGPU1 last seen running, not rechecked: Nightwatch816b1393f18c891ba8a8de11,supervisor2183339/process2184735; /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1. See DEPLOYMENT_REFIT.md. No held-out deployment score; monthly manifest preserved.

Vast root@ssh1.vast.ai:24199,/workspace/HACKATON,RTX PRO4000. Initial hierarchy/direct grid completed2.93min, selected direct September5.352400. First24-trial HPO completed5.61min, selected hierarchy32/two-block September5.282777; selection4.518243. Audited models/results retrieved locally results/hierarchical_vast and results/hierarchical_hpo_vast. Prior monthly winner remains better; it used31 August selection days versus hierarchy15 plus16 calibration, so architecture comparison is not controlled.

50/90/95% conformal added: August1–15 selection,16–31 calibration;50% marginal coverage42.33%, no nominal guarantee under drift. Point/interval replay passed.

Lag HPO24/24 trials over0/1/3/7/14/28 days completed exit0 in4.63min,23:07:59 Zurich October16. None beat parent on August1–15 selection: hierarchical32/two-block original history retained, September5.282777. Final marginal50/90/95% coverage42.33/86.49/92.93%; intervals under-cover. All replays/CSV audits passed, max hourly error5.68e-14. Nightwatch10e4974b8b1d9737a60aac8e; remote /workspace/HACKATON/results/hierarchical_hpo_lags_vast; local results/hierarchical_hpo_lags_vast. HIERARCHICAL_PROTOCOL.md. No active Vast experiment remains; prior monthly manifest preserved. Next optional followup: improve calibration/generalization; no new training authorized by completion check.

September-only baselines verified 2026-10-16: 30 days/11,520 quarters; MAE hourly mean10.695366, linear7.540344, mean-corrected linear7.090321 (33.71% reduction). Same centre alignment/boundaries as original baselines; predictions independently cross-checked against daily metrics. Artifacts: results/september_baseline_metrics.json, evaluate_september_baselines.py.

Monthly baseline comparison (2026-10-16): mean-corrected interpolation has lowest MAE in all12 months; September is hardest and December easiest for all3. Plot results/monthly_baseline_mae.png (also SVG), exact values results/monthly_baseline_mae.csv; script plot_monthly_baselines.py. Aggregated saved daily errors with equal zone weights;363 days, September cross-check passed.
