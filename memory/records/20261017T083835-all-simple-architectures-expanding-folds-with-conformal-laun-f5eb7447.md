---
title: All simple architectures expanding folds with conformal launched
status: current
kind: experiment
recorded_at: '2026-10-17T08:38:35.985447+00:00'
sources:
- EXPANDING_SIMPLE_PROTOCOL.md
- expanding_simple.py
- simple_search.py
- simple_conformal.py
- audit_expanding_simple.py
- run_expanding_simple.py
experiment_id: null
supersedes: []
---

Authorized2026-10-17: after asking conformal prediction, user asked test all with expanding window method. Scope retains complete224-grid for all four new families (linear/additive/MLP/CNN), plus5richridge regularizers and interpolation per fold. Eight chronological expanding folds (train3–10months,testsFeb–Sep) plus8fixedSeptember folds (trainlast3–10months endingJuly). Total3584gradient-trained candidates +80ridgefits +16 interpolation controls=3680 evaluations. Selection days1–15 of next month; calibrationdays16–end; followingmonth test. No calibration-based configuration/epoch selection. Historical prior campaign inspection means this is developmental even with clean within-campaign split. Causal history may extend before fixed-history parameter-fitting window, matching earlier protocol.

Conformal: absolute errors byzone/quarter(16groups), zone/hour max4quarters, whole-day max384quarters; nominal50/90/95%, corrected rankceil((n+1)c),infinityifrank>n. Finite95% day interval impossible with<=16calibration dates. Point means conserved; interval bounds coordinate envelopes, not mean-conserving realizations. No online recalibration/adaptive scale/point-MAE gain claim. Reference https://arxiv.org/abs/2202.13415 checked for dependence/drift caveat.

Remote root@93.91.156.85:55340,/workspace/HACKATON; RTX5090 UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9. GPU idle preflight; prior8worker concurrency benchmark reused. Nightwatch034d5133ebc5e076af6aed88 started1792226250.7036138 (10:37:30Zurich),initialETA20min due10:57:30; heartbeat_expected:true/timeout180s. Worker2088/runner2089; log/root/.local/share/nightwatch-hackaton/034d5133ebc5e076af6aed88.log. Wrapper runs conformalrankselftest,smoke,auditsmoke,full,auditfull; actualsuccess/crash reporting enabled.

Launch checks: smoke2folds(firstexpanding,firstfixedSeptember),8two-epoch configurations/fold covering4families×2features,plusridge/interpolation=20candidatechecks, independent audit passed maxmean5.684e-14. Fullparent2214 observed preparing withnoGPUallocationyet; no completion claim. Smokeoutputs results/expanding_simple_5090_smoke; fulloutputs results/expanding_simple_5090. Per-fold model.pt/predictions,chosenfamilymarginalintervalCSV/NPZ,andallcoveragelevels; learning_curves.csv, metrics.json,independent_audit.json,overall/per-zonePNG/SVG. Sourcecompiledlocally. Nextverifyfullworker/activity,thenendtrackedrunwithoutpolling; onreturnverifyterminalstatus/retrieveallartifacts/visuallyinspectplots. Original224resultsanddeploymentpreserved.
