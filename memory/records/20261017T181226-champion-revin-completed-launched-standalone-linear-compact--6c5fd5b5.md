---
title: Champion RevIN completed; launched standalone linear compact RevIN and explained
  features
status: current
kind: experiment
recorded_at: '2026-10-17T18:12:26.952980+00:00'
sources:
- results/champion_revin_20261017/audit.json
- results/champion_revin_20261017/summary.csv
- campaigns/linear_compact_revin_20261017/PROTOCOL.md
- results/linear_compact_revin_20261017/launch.json
- campaigns/monthly_ratio_20261017/experiment.py
- campaigns/monthly_quantiles_20261017/trainq.py
experiment_id: null
supersedes: []
---

User requested ALSO linear compact RevIN and explanation of features/model workings. Verified champion job cbf0b93e1888ddd4d52e0d11 completed all51freshfits/eightfolds, auditpassedzero replay error/hourlymeanmax1.14e-13; original control ratio0.7079682868 exactly recovered. RevINratio0.7251332599,pooledMAE4.2062893,90coverage92.4598%,width23.61079 versus originalMAE4.1031841,coverage90.5872%,width18.83545. Downloaded summary/audit/completion/report and full_day_normalized/full_day_eur_mwh.png, normalizedfigure visually inspected. Archives remain remote; no deployment promotion. New standalone campaign linear_compact_revin_20261017 freezes exact selected fixed_linear_compact_ratio source recipes from owning per-fold selection.json:15selected checkpoints,8folds,source ratio0.7245525295/MAE4.2017797. Fresh RevIN refits, prior stopping, freeze original lr/seeds/single-or-seedmean weights. Uses successful champion audited normalized-cache hash and same RevIN/physical-loss/audit/plot protocol; parentcompletion gated. Nightwatch94e67d8959b330d22189027a worker15570 on root@93.91.156.85:55340 /workspace/HACKATON, log/root/.local/share/nightwatch-hackaton/94e67d8959b330d22189027a.log;remote results/workspace/HACKATON/results/linear_compact_revin_20261017. Edge checks/cachepreparepassed and actualworkerverified; ETA10min actualsuccess/crash ownerTelegram. Requiredfirstlast2checkpointsmoke/full15replay,CQR audit,originalratio recovery,full-day plots/report/archive. Features verified from frozen owning feature builders:194=18currentphysical/calendar+9priorcontrast+63sevendayinterpolationerror signed/abs/availability+4causal nearestcurve quarter residual estimates+96knownfourzonedailyhourlyprices+4zoneonehot. Linearcompact head194->43 (3zero-sum median coefficients+40positivegap parameters); affine core8385parameters plus2RevINaffine. Median hourcontrasts early/late then withinfirst/secondhalf preservehourmean. Gaps softplus cumulativeordered11quantiles, originalcurrency MAE+pinball with fittingmonthratio weights;CQR prior16-end. MLP2x32,richCNN2temporalconv16 width5,2Djointzonepastquarter/hourgrid,additivehinge features;champion convexquantilefunctionblend. No currentquartertargets used asinputs. New protocol/launchscripts indexed; unrelated work preserved.
