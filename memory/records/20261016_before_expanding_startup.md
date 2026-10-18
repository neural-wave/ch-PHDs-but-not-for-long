# HACKATON project memory

Objective (2026-10-16): minimize quarter-hour MAE with official inputs only; no pretrained/external/October prices. Four zones jointly; hourly means within0.0001 EUR/MWh. Train Oct–July302 days, August31 validation, September30 development test(previously viewed). Data363 days/139,392 quarters.

Prior monthly winner monthly_joint_blend_gpu1: August4.245329, September5.091209;11,520-row serialized/CSV/causal audit passed. Evidence results/monthly_joint_blend_gpu1/audit.json, results/monthly_best.json; results/experiments_summary.html. RevIN wins2/4 pairs, no broad superiority. Original context: memory/records/20261016_before_hpo_lags_startup.md.

All363 frozen deployment refit on AndoGPU1 last seen running, not rechecked: Nightwatch816b1393f18c891ba8a8de11,supervisor2183339/process2184735; /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1. See DEPLOYMENT_REFIT.md. No held-out deployment score; monthly manifest preserved.

Vast root@ssh1.vast.ai:24199,/workspace/HACKATON,RTX PRO4000: hierarchyHPO winner Sept5.282777; lag0/1/3/7/14/28 search did not improve selection. Completed with replay/CSV audits.50/90/95% marginal coverage42.33/86.49/92.93%. Neural hierarchy won2/4 matched pairs; mean5.334665 vsdirect5.330407: no systematic quarter-MAE gain demonstrated.

User now prefers interpretability. Cached audit results/simplicity_audit.json: August-selected singleMLP Sept5.234445, rich ridge5.294221; ensemble gains2.74% and3.83% respectively. Final extensions over base ensemble gain0.67%. INTERPRETABLE_PROTOCOL.md.

New compact32-candidate linear/additive comparison completed8.51s onVast; Nightwatch18149e4a67cad97f5cf79dbd exit0. Selected27-input additive model Sept5.688563; readable model.json/coefficients.csv and audited metrics at results/interpretable_vast. Same302train/Aug1–15selection/16–31cal. Direct vsorthonormal hierarchical ridge equivalent(maxdifference7.8e-12). No active Vast campaign; prior manifest preserved. Next optional simplification: standalone rich ridge or compact additive depending acceptable accuracy loss. Detailed prior context memory/records/20261016_before_simplicity_startup.md.

Baseline chart follow-up (2026-10-16): monthly MAE split by anonymized ZONE_0–ZONE_3; official package does not identify countries. Mean-corrected interpolation wins all48 month/zone comparisons. Artifacts results/monthly_baseline_mae_by_zone.png/.svg/.csv; script plot_monthly_baselines_by_zone.py. Zone averages reproduce original monthly chart within1e-10. No model/job changes.
