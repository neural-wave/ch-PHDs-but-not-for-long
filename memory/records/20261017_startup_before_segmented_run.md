# HACKATON project memory

Objective: reconstruct four-zone quarter-hour prices from official inputs, preserving hourly means within 0.0001 EUR/MWh. No external/pretrained/October prices; history strictly earlier. Train October–July (302 days), select August (31), evaluate September (30; development, not blind). Zones anonymized.

Verified 2026-10-17: selected monthly_joint_blend_gpu1 September MAE5.091209 vs interpolation7.090321. Offline363-day deployment refit audited; no external submission. DEPLOYMENT_REFIT.md; results/COMPLETION_AUDIT.md. Simple family linear/additive/MLP/CNN MAEs5.209794/5.255107/5.343947/5.222494; August selected CNN. results/simple_5090. Hierarchy HPO5.282777; lag HPO retained parent. Quantile16-fold median5.226105; results/linear_quantiles_vast.

Error plots: results/error_hotspots_20261017, plot_error_hotspots.py. Joint blend hardest zone ZONE_1 MAE6.7331; September5 worst day-zone13.6688. Outer quarters and evening hours hardest; worst5% points contribute25.58% absolute error. Largest point error125.25 on September22 ZONE_0 hour20 quarter4. Descriptive frozen-prediction analysis only. Plots mark first/last hourly quarters with circles/squares and hourly shading.

Four-frequency study complete: Fourier4/free-phase8/PCA4/PCA8/PCA16 predictor MAEs6.210906/5.977431/5.827434/5.887884/5.887545; oracle diagnostic only. FOUR_FREQUENCY_PROTOCOL.md; results/four_frequency_20261017. Remote root@93.91.156.85:55340 /workspace/HACKATON; Nightwatch dcae73b185fbc7a157711e13 completed exit0, log /root/.local/share/nightwatch-hackaton/dcae73b185fbc7a157711e13.log.

User prioritizes interpretability, analog residuals, difficult-day robustness. INTERPRETABLE_ROBUSTNESS_PLAN.md; INTERPOLATION_HISTORY_PROTOCOL.md; results/feature_importance_20261017. Ordered-history ablation, SSM and seasonal segmented-frequency extension remain untested. Expanding/conformal campaign034d5133ebc5e076af6aed88 needs fresh final audit. Prior detailed context preserved in memory/records/20261017_startup_before_error_hotspots.md and earlier records. Next: audit expanding results; diagnose within-hour spikes using plots.
