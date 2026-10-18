---
title: Focused simple parallel 2D CNN improvement campaign launched
status: current
kind: experiment
recorded_at: '2026-10-17T11:31:46.053986+00:00'
sources:
- TUNED_PARALLEL_CNN_PROTOCOL.md
- tune_parallel_cnn.py
- audit_tuned_parallel_cnn.py
- results/feature_importance_20261017/ANALYSIS.md
experiment_id: parallel_2d_tuning_20261017
supersedes: []
---

2026-10-17 user requested ways to improve2D from5.4311 toward5.1–5.2, preserving simplicity. Implemented42fits: sevenarms(control8,width16,clock,residual,analog7,analog90,context) x twoLR.001/.003 x three seeds17/43/71. Shared parallelrawQ/H grid and2x8encoder retained. Newfeatures: hour/yearsin/cos; four extra paired-residualzonechannels containing historical interpolationerror and historicalhourly-queryhourly; fixedcausalnearest-curve residualsummary added4latechannels. Analog7/90 expands retrievalonly; CNNstill7days. Context arm fusionwidth5time, all otherpointwisefusion. No attention/U-Net/ensemble/previousmodelblend. Max600epochs/patience60allarms vs earlier300/30; reruncontrolparitygate isolates changes. Select onAugust only, freeze eacharm+overall beforeSeptember inference. Septemberdevelopment target notguaranteed, no adaptiveSeptemberselection. Prior frozenfeatureimportance is motivation, not evidence improvements will transfer. RemoteliveidleRTX5090 root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python; fourworkers. Healthpassed,currentNightwatchCLIcopied; run038fce95c7be0dda153059b8 worker5203, log/root/.local/share/nightwatch-hackaton/038fce95c7be0dda153059b8.log, initialETA15min, heartbeat_expectedtrue180s. Initialrunning verified; wrapperprepare/smoke/full/independentaudit+filterplots, actualsuccess/crashalerts. Sourcecompiledremote. Numericalresults pending. Priorstartup preservedmemory/records/20261017_startup_before_2d_tuning.md.
