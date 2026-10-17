---
title: Expanding monthly backtest and fixed-September learning curves launched
status: current
kind: experiment
recorded_at: '2026-10-16T21:54:27.681442+00:00'
sources:
- EXPANDING_PROTOCOL.md
- train_expanding.py
- plot_expanding.py
- run_expanding.py
- hierarchical_models.py
experiment_id: null
supersedes: []
---

2026-10-16 user asked how neural hierarchy helps interpretability, authorized training through some month/validation next/test next-next with expanding training, then explicitly requested graphs of performance versus larger training sets. Explained physical output contrasts a (half-hour),u,v (within-half quarter differences): H+a+u,H+a−u,H−a+v,H−a−v. These explain output structure, not opaque neural input-to-contrast mappings or causal feature effects.

Implemented16-fold dual study on existing Vast root@ssh1.vast.ai:24199,/workspace/HACKATON,RTXPRO4000 (idle verified before launch). Eight expanding folds starttrainOct–Dec2025/valJan/testFeb2026, then grow training end monthly throughJuly/valAugust/testSeptember. Second8-fold learning curve holds latesttrainJuly,valAugust,testSeptember fixed while adding older trainingmonths: last3May–July throughlast10Oct–July. This controls test-season and most-recent-example changes; amount of data still varies in distribution as older months are added. No forced monotonic curve or automatic claim that more history helps. Actual official days retainDSTexclusions.

Per-fold selection days1–15 of validation month;16–end conformal calibration50/90/95, samepreviousdayobserved-history rule, no onlineparameterfitting or validationrefit. Normalization/parameters train dates only. Fresh fit every fold. Neural familiesdirect/hier4configseach: widths32/64,seeds17/43,oneblock,dropout.1,lr.001,batch32,weightdecay.01,max160epochspatience20. SamequarterMAEloss with NO auxiliaryhalf-hourloss, correcting previous confound; headparametercounts still differ.128fullneuralfits total16folds. Add single richridgealphagrid1/10/100/1000/10000 and interpolation; no ensemble. Freeze familychoices beforecal/testmetrics; savewidth/seedpairedcomparisons. SourceSHAexpectedf71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c. Tests enforcefoldchronology,causalfeaturemutation,trainonlynormalization,finite/hourmeanconstraints,selectedserializedreplays.

Graphs planned: expanding_learning_curve.png/.svg withtestmonthlabels,fixed_september_learning_curve.png/.svg,per-zonecurves,hierarchy_benefit.png(stagecomparison),hierarchy_stages_example.png(firstSeptdayZONE0,selected10monthmodel). Rawlearning_curves.csv,metrics.json,per-foldselected.pt/predictionNPZ/stagecontrastNPZ. Smoke2folds2epochsinresults/expanding_smoke_vast beforefullresults/expanding_vast. Local syntax checks passed; runtime outputs pending.

Nightwatch47a057b83338d7af7dfb5ceb registeredrunning,heartbeatexpectedtrue180stimeout,worker2039; currentCLIalreadyexecutionhost. Log/root/.local/share/nightwatch-hackaton/47a057b83338d7af7dfb5ceb.log. Started23:53:31ZurichOctober16,ETA15m (00:08October17), not evidenceofcompletion. WrapperinstallsMatplotlib,runsGPUpreflight,smoke,full,andplots; actualfailure/completionTelegramalerts enabled. Next verify smoke/fullprocess and actualterminalstate; retrieve visuallyinspectfinalplots beforepresenting. No previous results or deployment manifests altered.
