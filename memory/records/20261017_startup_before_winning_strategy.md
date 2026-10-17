# HACKATON project memory

Objective: reconstruct four-zone quarter-hour prices using official data only; preserve hourly means within0.0001 EUR/MWh. No external/pretrained/October prices. Strictly prior observed history allowed. Main protocol: October–July302 fitting days, August31 selection, September30 development test; September previously inspected, never a blind-test claim. Zones anonymized.

Verified sources2026-10-17: selected monthly_joint_blend_gpu1 SeptemberMAE5.091209 vs corrected interpolation7.090321 (~28.2% lower). Separate all363-day refit and22-component offline submission completed/audited per DEPLOYMENT_REFIT.md and results/COMPLETION_AUDIT.md; no held-out accuracy for refit, no external submission. Prior60-day scores are historical.

Completed224-model RTX5090 simple search: SeptemberMAE linear5.209794,additive5.255107,MLP5.343947,CNN5.222494; August overall winnerCNN. results/simple_5090. Hierarchy HPO5.282777; lag HPO retained parent. Both use15selection/16calibration days. MAE-only16-fold expanding study complete at results/expanding_vast, starts3trainingmonths. Seven-day ordered signed/absolute interpolation-error inputs implemented/verified2026-10-17, optional compact7; INTERPOLATION_HISTORY_PROTOCOL.md. Training ablation pending; no improvement claim.

Feature reliance verified2026-10-17: results/feature_importance_20261017/ANALYSIS.md. Frozen simple linear/additive rely most on interpolation-error history, especially similar-curve analogs; MLP slopes/curves; CNN daily curves. Exact2/7-day lags weak. Permutation reliance, not retraining benefit; monthly blend untested.

Quantile16-fold: final medianMAE5.226105; learned50/90/95coverage52.86/90.90/95.01%; CQR54.62/93.39/96.81%. results/linear_quantiles_vast; LINEAR_QUANTILES_PROTOCOL.md. Coverage empirical.

Live2026-10-17: all-simple expanding/conformal campaign active root@93.91.156.85:55340,/workspace/HACKATON; parent2214,worker2808 fixed_september_4m. Nightwatch034d5133ebc5e076af6aed88; log/root/.local/share/nightwatch-hackaton/034d5133ebc5e076af6aed88.log. Partial metrics exist; final independent audit absent. EXPANDING_SIMPLE_PROTOCOL.md;3680 evaluations/16folds. Next verify completion, retrieve/audit outputs and inspect plots. Preserve all prior artifacts. Previous startup archived memory/records/20261017_before_project_overview_startup.md.
