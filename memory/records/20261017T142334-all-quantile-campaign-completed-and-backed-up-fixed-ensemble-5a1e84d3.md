---
title: All-quantile campaign completed and backed up; fixed ensemble mixing and model
  specialization audited
status: current
kind: result
recorded_at: '2026-10-17T14:23:34.937576+00:00'
sources:
- results/monthly_quantiles_20261017/REPORT.md
- results/monthly_quantiles_20261017/audit.json
- results/monthly_quantiles_20261017/summary.csv
- results/monthly_quantiles_20261017/backup_verification.json
- results/monthly_quantiles_20261017/pooled_confidence_summary.json
- campaigns/monthly_quantiles_20261017/analyze_ensemble.py
- results/monthly_quantiles_20261017/ensemble_analysis/REPORT.md
- results/monthly_quantiles_20261017/ensemble_analysis/audit.json
- results/monthly_quantiles_20261017/ensemble_analysis/visual_verification.json
experiment_id: monthly_quantiles_20261017
supersedes:
- memory/records/20261017T140723-quantiles-made-the-default-all-quantile-monthly-ratio-study--75c91109.md
---

Verified17October2026: Nightwatch5d8339ecf415745b4844ddb1 completedexit0 at14:12:25UTC/16:12:25Zurich. All384neural fits/eight expanding folds finished. Independent audit passed384candidate rescoring,96selected checkpoint replays exactly0error, maxhourlymeanerror1.1368683772161603e-13,92544exportedquarters, quantile order/nonzero widths, train-only statistics/weights, validation choices, quantile blends, CQR ranks/corrections/nesting/coverage, confidencepercentiles/bins and CSVidentity. No active job remains for this campaign. Hostroot@93.91.156.85:55340,/workspace/HACKATON; log/root/.local/share/nightwatch-hackaton/5d8339ecf415745b4844ddb1.log. Complete local archive results/monthly_quantiles_20261017_completed.tar.gz verified:1312988267bytes,SHA256bc677aaa552cadbf233eb792f2d391927c11de0b3eaf335411f68364f31d10cb againstremote completionreceipt. Userdefault alwaysquantiles retained; catastrophe/risk-classifier branch remains set aside. No deployment change.

Full quantile scores(meanmonthlyMAE/correctedinterpMAE,pooledMAE,calibrated90%coverage,width): blend_uniform0.720494/4.176488/92.301%/20.7515EUR; validation_selector0.721285/4.181172/92.579%/21.4882; blend_ratio0.721680/4.182921/92.730%/21.9636. Bestsingle compactlinear_ratio0.730386/4.234448. Ordinary compactlinear0.730864/4.237331. Previouspointblend0.720436/4.177390, so quantile point accuracy verysimilar, not a significance claim. All months remain developmental. Confidence export for validation_selector: narrow23099quarters MAE2.331793, middle40824MAE3.599795, wide28621MAE6.502997; corresponding baselineMAE2.742205/4.753564/9.808349, so width also tracks baseline difficulty. These are relative width groups, not correctness probabilities. predictions.csv.gz retains11quantiles/calibrated90%bounds,width,percentile,label; per-modelNPZs have all50/80/90/95%bounds.

User then asked how ensembles mix, whether models specialize in certain parts/types, and about interpreting the router. Current system has no input-dependent router. Each fold selects fixed global convex weights on preceding-month days1–15, then applies the same weights to every zone/hour/quarter and all quantilelevels through the test month. Quantile functions are averaged; blends are separately calibrated. Candidate choices are individual model, rich-linear pair withweights.25/.5/.75, or equaltop3validation models. 'Uniform'/'ratio' name the training objective, not the mixture weights. Ordinary blend: Feb equalCNN/MLP/additive; Mar50%richlinear/50%MLP; Apr equalMLP/analogCNN/CNN; Mayequalrichlinear/MLP/CNN; Juneequalcompactlinear/analogCNN/CNN; JulyequalMLP/analogCNN/CNN; Augequalrichlinear/analogCNN/CNN; Sepequalrichlinear/MLP/analogCNN. All weight identities independently reconstructed for every raw quantile with0error.

New posthoc specialization diagnostics ran on research host without fitting or changing any predictor. Group errors by zone,hour and known hourlycurve(flat/rising/falling/peak/trough), plus ensemble uncertainty. Flat thresholds are fitting-history25th-percentile maximum neighboring ramp. Conditional score divides by interpolation MAE on the identical context subset and averages monthly ratios; do not confuse these with unconditional overall ratios. Clearest specialization: analog2DCNN winsZONE_2 in7/8months for BOTH objective variants; zone2ratio0.829255ordinary/0.823584ratio-training. Compactlinear bestoverall, oftenbestZONE_0/3; richlinear bestZONE_1 aggregate. Curve rankings vary considerably bymonth; no stable causal-specialist claim.

Complementarityordinary: actualblendratio0.720494 versusweightedmeanconstituentabsolute-errorratio0.747499, cancellationgain0.027005. Blendbeatsall6experts on5.42%wholezonehours. A hindsightoracle choosinglowesttruefour-quarter error expert for eachwholezone/hour reaches0.556926(ordinary)/0.560528(ratio). These are NOT deployable routing scores: oracle readsobservedquarters, and no gatewastrained. Actualinterpretablegate wouldfitonlyearlier out-of-sampleexpert errors usingknownzone/hour/slopes/spreads/history/uncertainty/disagreement, preservewholehourquartermeans and recalibratefinalgatedquantiles. This remains unperformed.

Interactive view saved /Users/miguelnogales/.codex/visualizations/2026/10/17/01a14a1b-61ed-7000-a5ed-a18641bdccd3/ensemble-specialization.html. Defaultoverall0.7205 andZONE2analog0.8293 verified inbrowser; monthfilter1weightrow,7scorerows, objectivechange, desktop/360px layout and no consoleerrors verified. Under1MB. Data/weights/conditionalCSV/audits in results/monthly_quantiles_20261017/ensemble_analysis/. Original recovery archive predates this posthocanalysis; analysis source/data separately saved locally and remotely.

## Verbatim previous startup

# HACKATON project memory

Objective: official four-zone quarter-hour reconstruction; hourly means within 0.0001 EUR/MWh. Official data only; no external/pretrained components, October evaluation prices or inference-time refitting. Quarter history strictly prior. Developmental evaluation; anonymized zones; prefer interpretable CNNs.

Adopted VALIDATION_PROTOCOL.md: eight expanding folds, three initial training months, next-month days 1–15 selection/16–end optional calibration, following-month test (February–September). Fresh train-only fitting/scaling; frozen test weights. Primary score: equal-month mean(MAE/corrected-linear-interpolation MAE); retain pooled MAE. One inner selection block, not full nested validation.

Verified 2026-10-17: Riccardo commit 1ed1423c5fa05586457347ab34e69b9bcfb23031 compared with nine existing pipelines on 241 days/92,544 targets. Our linear remains first: ratio 0.738303/MAE 4.273508; Riccardo reduced 0.796834/4.615017 (rank 9/11), raw 0.803239/4.658695 (rank 10/11). Linear wins all eight months. Original 5.087239 August–September score uses joint selection/all-year weight scorers; fresh train-only weights used here. Audits passed: 208 scorer fits/64 serialized models. Results/checkpoints retrieved: results/riccardo_comparison_20261017/REPORT.md. Prior nine-method evidence: results/normalized_validation_20261017/REPORT.md.

September-only blend 5.091209, analog90 CNN 5.177024; Riccardo reduced 5.543063. Blend uses full August; others first15. Newer CNN/blend multi-month estimates absent. Offline 363-day deployment refit audited; no submission change. Exploratory FlowState control 5.454209 versus FlowState+CNN 5.461774; no benefit. FlowState archive verified; from-scratch SSM untested.

Execution: root@93.91.156.85:55340,/workspace/HACKATON. Nightwatch 74e4190bed4235d2126ac27b completed exit0; no active comparison/FlowState job. Detail and verbatim prior startup: memory/records/20261017T141346-riccardo-branch-cnns-completed-under-adopted-eight-fold-vali-0484b8f2.md. Next: retain deployment; use adopted scoring for future tuning.
