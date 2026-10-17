---
title: Project-wide summary and live expanding-search status on 17 October
status: current
kind: summary
recorded_at: '2026-10-17T08:45:28.900379+00:00'
sources:
- results/COMPLETION_AUDIT.md
- DEPLOYMENT_REFIT.md
- results/september_baseline_metrics.json
- results/simple_5090/metrics.json
- results/expanding_vast/metrics.json
- results/linear_quantiles_vast/metrics.json
- EXPANDING_SIMPLE_PROTOCOL.md
experiment_id: null
supersedes: []
---

Reviewed completed campaigns for user-requested overview on 2026-10-17. Best monthly August-selected evaluation ensemble remains September MAE5.09120882 vs corrected interpolation7.09032123 (~28.2% reduction). Ensemble uses302 fitting days; final all363-day submission refit is separately completed and audited according to DEPLOYMENT_REFIT verified-final-state and COMPLETION_AUDIT, correcting startup's stale 'not rechecked' uncertainty. Submission was prepared locally, not sent to jury. Historical60-day protocol scores must not mix with September-only metrics.

224-model simple search completed: family August winners September linear5.209794, additive5.255107,MLP5.343947,CNN5.222494. Overall August winner CNN; linear has lower descriptive September score. Hierarchy HPO5.282777 uses15selection/16calibration August days; lag HPO retained parent. Expanding MAE-only study completed16folds; February–September target-weighted MAE ridge4.373205,direct4.433215,hierarchical4.419719,interpolation5.814817. These are not September-only scores. Fixed-September history lengths distinguish data size from seasonal change. Quantile16fold study completed/audited; final September medianMAE5.226105 and learned50/90/95 bands coverage52.86/90.90/95.01%; widen-only CQR54.62/93.39/96.81%. Temporal coverage empirical.

Live read-only SSH check on root@93.91.156.85:55340: run_expanding_simple PID2089, parent2214 and fixed_september_4m worker2808 active; last log trial98/224 in current fold. /workspace/HACKATON/results/expanding_simple_5090/metrics.json exists but independent_audit.json absent; no final completion claim. Nightwatch034d5133ebc5e076af6aed88; log/root/.local/share/nightwatch-hackaton/034d5133ebc5e076af6aed88.log. Campaign plans3680 candidate evaluations across16folds. Next retrieve and audit final artifacts/plots after completion; preserve deployment and prior manifests. No new experiment launched.
