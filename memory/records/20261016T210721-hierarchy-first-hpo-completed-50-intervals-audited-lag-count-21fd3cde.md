---
title: Hierarchy first HPO completed, 50% intervals audited, lag-count HPO launched
status: current
kind: experiment
recorded_at: '2026-10-16T21:07:21.818321+00:00'
sources:
- results/hierarchical_vast/metrics.json
- results/hierarchical_hpo_vast/metrics.json
- HIERARCHICAL_PROTOCOL.md
- hierarchical_features.py
- train_hierarchical.py
- predict_hierarchical.py
- run_hierarchical_hpo.py
experiment_id: null
supersedes: []
---

2026-10-16 user asked why35m, authorized HPO, added50% quantile, then requested amount of lags. Live Nightwatch confirms initial eight-candidate run5c9227f73977427b05cb824b completed in175.79s (2.93min);35m had been conservative setup allowance. Initial August1–15-selected direct64 seed43 candidate7: selectionMAE4.5505071356, September5.3523995409. Serialized point/interval replay maxerror2.130866e-5.

First randomized24-trial HPO3af6c0409c42daef1cdd901b completed in336.39s (5.61min); summed actual training257.70s. Eighteen hierarchical/six direct, widths32/64/128, depth1/2, dropout, log-uniform lr/weight-decay, half-hour auxiliary weight, batch16/32/64, flat/interpolation anchor; max160 epochs/patience20. Original selected model included frozen reference. Selection exclusively Aug1–15; calibration Aug16–31 remains separate. Winner candidate1 is hierarchical32,2blocks,dropout0,lr0.0009119043,weightdecay0.0034984067,aux0.1,batch64,flat anchor,seed114831,bestepoch66 (86run). SelectionMAE4.5182432717; August descriptive4.3836934611; September5.2827771386; half-hour4.2423023966. It improves first-run SeptemberMAE but does not outperform historical monthly joint blend5.091209; do not replace manifest. First HPO's point/interval replay maxerror9.834766e-6. Metrics/testNPZ and winning checkpoints retrieved locally under corresponding results directories.

Added50% marginal/hour-joint/day-joint conformal bands from held-out corrected0.5 absolute-error quantile, not a separately trained median point predictor. Initial point model unchanged; 50% point/interval full replay passed. For first HPO50% marginal September coverage0.4232638889, mean width5.2347613936 EUR/MWh: below nominal50%, so no nominal-coverage claim. Hour-joint50% radius bounds have marginal0.6682291667 (different target); day-joint50% actual day coverage0.4666666667, width85.6653735733. Time dependence/drift caveat retained.

User lag request arrived during first HPO. An attempted stop found worker1523 already exited: first HPO completed successfully, was NOT interrupted. Lag24-trial extension now runs with exactly0/1/3/7/14/28 prior daily quarter-residual lags,4trials per count(3hierarchy/1direct), replacing all legacy fixed/rolling/analog-history features;333+4N current/calendar+history features. Causal history/no-history independence checks passed for every count, and one/two-block architecture conservation checks passed. Original HPO's August-selected winner retained as389-feature reference. Separate train-only normalization per profile; no calibration/September-based HPO decisions. Lag grid declared before reading first-HPO September result.

New Nightwatch10e4974b8b1d9737a60aac8e running with live heartbeats,worker1686; remote /workspace/HACKATON/results/hierarchical_hpo_lags_vast; log /root/.local/share/nightwatch-hackaton/10e4974b8b1d9737a60aac8e.log. Started23:03:21 Zurich; ETA8m (23:11), not evidence of success. Last verified18/24 trials complete and wrapper alive. Next verify terminal status, retrieve lag results, compare August-selected winner and coverage/widths. Existing Ando job not rechecked.
