---
title: 'Actual six-expert normalization comparison completed: selector ratio0.708371
  versus original0.720494'
status: current
kind: result
recorded_at: '2026-10-17T16:05:23.209221+00:00'
sources:
- campaigns/ensemble_normalization_20261017/PROTOCOL.md
- campaigns/ensemble_normalization_20261017/train.py
- campaigns/ensemble_normalization_20261017/audit.py
- campaigns/ensemble_normalization_20261017/run.py
- results/ensemble_normalization_20261017/REPORT.md
- results/ensemble_normalization_20261017/audit.json
- results/ensemble_normalization_20261017/metrics.json
- results/ensemble_normalization_20261017/insights.json
- results/ensemble_normalization_20261017/mixed_composition.json
- results/ensemble_normalization_20261017/launch.json
- results/ensemble_normalization_20261017/backup_launch.json
- results/monthly_quantiles_20261017/summary.csv
- results/zone_weights_20261017/INTERPRETATION.md
experiment_id: ensemble_normalization_20261017
supersedes: []
---

User annotated the earlier compactCNN/S4D/S5 normalization table asking about the other models and recalled score0.72. Scope correction: that table omitted the actual stronger six-expert ensemble and cannot determine normalization effects there. Original ordinary-training blend0.7204941118854624, original objective selector0.7212852815039343; later zone-shrinkage0.7189066090196203 is separately posthoc-selected. Read original owning scores/composition and verified audits. Extended the original authorized normalization task to its actual six original experts and original blend menu, rather than substituting weaker SSM controls.

Scientific extension completed and independent full audit passed: Nightwatch32cd0b9949f2486cbac56c7e exit0at17:31:47Zurich17October2026. Host root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python; log/root/.local/share/nightwatch-hackaton/32cd0b9949f2486cbac56c7e.log.192new fixed-scaling fits +192exactly matched reused standardized controls;384checkpoint replays, maximum discrepancy0. First/last-fold24twoepochfits smoked/audited before full training. Original standardized checkpoints/states/predictions verified identical and blend score reproduced exactly0.7204941118854624. Network parameters matched per family. Same rates.001/.003,seeds17/43,architecture,raw feature snapshot,quantile output scale20,MAE+pinball,AdamW,batch32,epoch/patience,clip5. Ordinary/uniform training objective only; no ratio-training confound. Fixed skips fitted z-scores but retains engineered/100levels,/20differences and same[-15,15]clip; compact zone onehot and fitting-only additive knots retained. This is an input preprocessing/optimizer/clipping comparison, not a claim about exact RevIN or normalization in general.

Eight expanding February–September developmental tests, preceding first15days select,16–end calibrate, no selection/calibration refit, strictly prior observed quarters, known hourly curves, frozen test parameters. Eleven ordered quantiles with conservedq50 and independentCQR50/80/90/95. Family/config, pure/mixed blends and selector freeze before test scoring. Pure arms use original menu (individuals,richlinear pairs.25/.5/.75,equal top3). Mixed selects preprocessing separately per family using preceding validation only, then same menu. All declarations written before outcomes. No deployment or submission update.

Mean monthly MAE/interpolation ratios standardized→fixed:richlinear0.735546633→0.728019913;compactlinear0.730863726→0.725706687;additive0.737004101→0.730141708;MLP0.746533453→0.734968912;rich temporalCNN0.768762201→0.738977995;analog2DCNN0.736777522→0.727906192 (exact full values in metrics.json). Relative gains respectively1.0233%,.7056%,.9311%,1.5491%,3.8743%,1.2041%. Fixed wins selected months8/8,8/8,7/8,8/8,7/8,6/8; same-family/lr/seed wins31/32,31/32,29/32,28/32,30/32,19/32. These are descriptive dependent backtests, not independent significance samples.

Pure fixed blend ratio.7103591550594415,pooledMAE4.1183453072,90%coverage90.0231%,width18.16156;1.4067%relative ratio gain versus original.720494,7/8months improved. Mixed blend and validation selector both.7083710434569991,pooledMAE4.1067849060,90%coverage90.3495%,width18.342485;1.6826%relative ratio gain,all8months improved,width11.6090%lower than original20.751523. Selector actually chooses pure fixed in7folds and mixed inJune. Mixed weights use23fixed components of24: standardized analog2DCNN retained only inJune, coupled with fixedMLP/fixedtemporalCNN. Equal-weight top3 recipes throughout;MLPfixed present8months,temporalCNNfixed7,analog2DCNNfixed5,standardanalog1,and linear/additive experts earlier folds. All aggregate ratios/pooledMAE independently recomputed locally from tiny savedCSV summaries. No claim of blind future performance. Broader contemporaneous expanding-optimization study is separate (pretest selector.708892,bestposthoc.707968); its source/results preserved in current startup and not overwritten or attributed solely to normalization.

Scientific results/summaryCSV/audit/manifests and mixed composition retained locally under results/ensemble_normalization_20261017. Remote full recovery archive1555212494bytes,SHA2566acd7597146beaa22bcda64620f1af914ba42335f5b491641afe0bf368c35657. Local full archive backup running under detached Nightwatch4a3b428fe17b9775beee963f,worker12285,log/Users/miguelnogales/Documents/Codex/.nightwatch/workers/4a3b428fe17b9775beee963f.log; ETA18:08:04Zurich. Own durable subprocessscp and streamingSHA verifier; expected180s heartbeats and actual success/failure alerts. Live API confirmed running with subsequent heartbeat; transfer/log verified. Receipt will be results/ensemble_normalization_20261017/backup_verification.json. Do not claim localfullSHA verified until receipt. Scientific result already complete. Initial predecessor affineS4Djobd85b1d1b1b58a45af40c3a57 completedexit0 before this GPUwork started; no activejob interrupted. Prior current startup preserved verbatim in memory/records/20261017_startup_before_ensemble_normalization_results.md. Next on return: verifybackupreceipt; inspect normalization mechanisms only if requested. No additional model run is pending in this extension.
