# HACKATON project memory

Objective: reconstruct four-zone quarter-hour prices from official inputs; hourly means within0.0001 EUR/MWh. No external/pretrained/October prices; strictly prior observed history permitted. October–July302 fitting days, August31 selection, September30 development days; September is not blind. Zones anonymized.

Verified2026-10-17: monthly_joint_blend_gpu1 SeptemberMAE5.091209 versus interpolation7.090321. All363-day refit/22-component offline submission audited: DEPLOYMENT_REFIT.md, results/COMPLETION_AUDIT.md; no held-out refit score or external submission. Simple224-model search SeptemberMAE: linear5.209794, additive5.255107, MLP5.343947, CNN5.222494; August selectedCNN. results/simple_5090. HierarchyHPO5.282777; lagHPO retained parent. Expanding16-fold MAE study completed: results/expanding_vast.

Feature reliance: results/feature_importance_20261017/ANALYSIS.md. Similar-curve interpolation-error history dominates frozen linear/additive history reliance; exact2/7-day lags weak. Monthly blend untested; permutation is not retraining evidence. Ordered seven-day signed/absolute inputs verified, training ablation pending: INTERPOLATION_HISTORY_PROTOCOL.md.

Quantile16-fold medianMAE5.226105; learned50/90/95coverage52.86/90.90/95.01%; CQR54.62/93.39/96.81%, empirical only. results/linear_quantiles_vast; LINEAR_QUANTILES_PROTOCOL.md.

Last live check2026-10-17: expanding/conformal3680-evaluation campaign active root@93.91.156.85:55340,/workspace/HACKATON; parent2214,worker2808,fixed_september_4m. Nightwatch034d5133ebc5e076af6aed88; /root/.local/share/nightwatch-hackaton/034d5133ebc5e076af6aed88.log. Final audit absent. EXPANDING_SIMPLE_PROTOCOL.md. Next verify/retrieve/audit results and plots.

User direction2026-10-17: prioritize intrinsic interpretability, confidence-weighted analog residuals and difficult-day robustness. Concrete proposed study: INTERPRETABLE_ROBUSTNESS_PLAN.md; gates require causal out-of-fold evidence and validated fallbacks. Jury combines MAE with qualitative assessment. No new training launched. Details: memory/records/20261017T085943-winning-direction-recommendation-historical-analogs-robust-v-543372b4.md. Earlier startup preserved: memory/records/20261017_startup_before_winning_strategy.md.

SSM design remains untested; see memory/records/20261017T093103-s4-s4d-and-s5-feasibility-for-constrained-price-reconstructi-5f2782b9.md.

2026-10-17: user proposed four daily frequency weights. Fixed-basis mean-preserving residual model suggested; free phases need extra coefficients. Oracle approximation diagnostic recommended before training. No run launched. Previous full startup: memory/records/20261017_startup_before_four_frequency_proposal.md.
