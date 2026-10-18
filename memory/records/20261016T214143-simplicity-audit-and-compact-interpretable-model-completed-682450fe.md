---
title: Simplicity audit and compact interpretable model completed
status: current
kind: result
recorded_at: '2026-10-16T21:41:43.976640+00:00'
sources:
- results/simplicity_audit.json
- results/hierarchy_paired_comparison.json
- INTERPRETABLE_PROTOCOL.md
- results/interpretable_vast/metrics.json
- results/interpretable_vast/model.json
- results/interpretable_vast/coefficients.csv
- audit_simplicity.py
- train_interpretable.py
- interpretable_models.py
experiment_id: null
supersedes: []
---

2026-10-16 user prefers interpretable/simple model and asks incremental ensemble and hierarchy gains. Audited cached individual predictions on Ando (no refitting) with exact302train/31Augselect/30Sept split. Best overall single selected by August is net28: August4.3169085466, Sept5.2344449331. Ensemble Sept5.09120882 gains0.1432361131 EUR/MWh,2.7364% over selected single. August-selected ridge1000 Sept5.2942212791: ensemble gains0.2030124592,3.8346%. Other family-selected Sept values graph12 5.1199504066(gain0.5614%),cat9 5.1878966978(gain1.8637%),boost31_10 5.2515537728,forest5 5.3476567327,RevIN7 5.3288629919. Family choices never use September; graph is not the overall validation winner and not inherently interpretable. Base ensemble already5.1258003184, final layered extensions improve0.0345914984,0.6748%. No statistical-significance claim; September is developmental.

Initial width32/64×seed17/43 hierarchy/direct matched packages: hierarchy Sept wins2/4, selection wins1/4; meanSept5.3346645770 vsdirect5.3304074973, hierarchy0.0799% worse. Half-hour means4.2843638773 vs4.3064194318, hierarchy0.5122% better. Four pairs too small for broad claims; hierarchy also adds0.25 halfhour loss, so not head-only comparison. Prior HPO18hier/6direct unequal budgets cannot establish hierarchy gain. Existing direct forecasts already preserve hourly means by projection.

Implemented and completed isolated32-candidate compact model comparison on Vast (not local compute):18 current price/slope/cross-zone/calendar inputs or27 with previous-day/week/7day-mean contrast history; per-zone linear or additive piecewise-linear effect functions, flat/interpolation anchor, alpha1/10/100/1000. Same302train/Aug1–15selection/Aug16–31cal/Septtest protocol as neural hierarchy. Fit squared-error ridge, select quarterMAE. No ensemble. Winner additive_flat_history1_alpha10,27 inputs,1308 coefficients (three contrasts×four zones×109basis terms), selection5.0585558194, August4.8345545767, Sept5.6885633315. This much smaller interpretable feature set loses accuracy versus rich389-feature ridge5.2942. These two also differ in selection budget, so not controlled solely by feature count.

Every trial directly verifies that direct-quarter ridge and orthonormal hierarchical-contrast ridge with SAME features/regularization/loss are equivalent: maxdifference7.787548e-12. Causal-history tests, exact hourly conservation(maxerror1.136868e-13), serialized JSON replay(maxerror0),11520-row finite uniqueCSV audits all passed. Empirical conformal50/90/95 coverage41.7448/85/91.9705%; undercoverage persists. Nightwatch18149e4a67cad97f5cf79dbd completed exit0 in8.51s. Remote /workspace/HACKATON/results/interpretable_vast; log /root/.local/share/nightwatch-hackaton/18149e4a67cad97f5cf79dbd.log. Model.json,coefficients.csv,metrics,testNPZ retrieved locally. Existing monthly/deployment manifests preserved. Recommend one rich ridge as accuracy/simplicity compromise; compact GAM as clearer but less accurate alternative. No additional campaign pending.
