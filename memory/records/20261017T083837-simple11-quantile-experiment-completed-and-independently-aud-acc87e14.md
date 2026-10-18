---
title: Simple11 quantile experiment completed and independently audited across16 folds
status: current
kind: result
recorded_at: '2026-10-17T08:38:37.604391+00:00'
sources:
- results/linear_quantiles_vast/metrics.json
- results/linear_quantiles_vast/learning_curves.csv
- results/linear_quantiles_vast/interval_curves.csv
- results/linear_quantiles_vast/independent_audit.json
- results/linear_quantiles_vast/fixed_september_10m/model.json
- results/linear_quantiles_vast/fixed_september_10m/coefficients.csv
- LINEAR_QUANTILES_PROTOCOL.md
- audit_quantile_results.py
- plot_linear_quantiles.py
experiment_id: null
supersedes: []
---

2026-10-17 completion check requested. Live Nightwatch0bf41a85e0867de319a9e14a completed exit0,start1792225499.7288,end1792225689.1765:189.45s (~3min9s),10:28:09Zurich. Log /root/.local/share/nightwatch-hackaton/0bf41a85e0867de319a9e14a.log ends DONE quantilefolds/audits/coefficients/graphs;16/16folds complete. Remote /workspace/HACKATON/results/linear_quantiles_vast onroot@ssh1.vast.ai:24199. Allquantiletrainingcompleted; no active job fromthisrequest.

Final10month fixedSeptember model(302train/15select/16cal/30test) selectedlr.001,L2.001,epoch67/92:medianMAE5.2261054787,halfhourMAE4.1339832736,mean11pinball1.5055412227 EUR/MWh. ZoneMAE3.9567244702,6.8454343476,5.6023407999,4.4999222972. Modelisone389-feature linear layer43outputs,includingthree hierarchicalmediancontrasts;all11levels remain evaluated. Raw adjacent-quantilecrossing rate5.2096% across heldout evaluation rows beforefixedmedianclamp/rearrangement;finalquantilesordered. Coefficientmeaningappliestorawlinearfunctions;postprocessing documented. Independent JSONNumPy replaymaxerroracrossfolds1.136868e-13.

FinalSep learned/rearranged quantile interval coverages50/90/95nominal:52.8646/90.9028/95.0087%, widths8.678863/25.916554/33.997654 EUR/MWh. Widen-onlyCQR:54.6181/93.3854/96.8056%,widths8.960411/27.900813/36.749942. Median-residualCP stillundercoverage42.5087/85.9983/92.4826%,widths5.446200/20.108346/27.661774.30minmiddlestage medianCP43.3681/85.7292/93.0556%. Quantile-basedboundswellcalibratedmarginallyinthisSepstudy;CQRconservativeandwider,notfreeimprovement. Temporalcoverageempirical,notunconditionalguarantee.

FixedSep3→10months:medianMAE5.3779004704→5.2261054787;meanpinball1.5725155098→1.5055412227;notmonotonic. Expandingfoldschange testmonth,so first/lastMAE3.087666→5.226105 notanisolatedtrainingsizeeffect. PreserveexistingMAE-onlyreference andotherRTX5090searchcontext.

Retrievedlocalresults/linear_quantiles_vast:summarymetrics/CSVs/allPNG plusfinal10monthmodel.json,coefficients.csv,test_predictions.npz. Visuallyinspectedlearningcurveandintervalquality. Foundreferenceoverlayfilteromittedfamily,creatingduplicatedzigzags;fixedplot_linear_quantiles.py tofilterfamilyandlabelridgebaselinecorrectly,regeneratedwithoutretrainingandrefetched. Finalgraphsinclude fixed_september_learning_curve.png,expanding_learning_curve.png,fixed_september_interval_quality.png,expanding_interval_quality.png,all11_quantile_calibration.png.

Newindependentauditreadsall16foldpredictionNPZsandoriginalofficialtruth;passed16folds/192intervalcases. Checksdateidentity/uniqueevaluateddates,truthidentity,finiteordered11quantiles,medianidentity,recomputedMAE/allcoverage/allwidths,hourlymeanconservation. Maxmeanerror1.136868e-13,maxmetricdifference7.105427e-15. results/linear_quantiles_vast/independent_audit.json. Priorlearning-curveplotcodefixedonly;no model/manifests changed. Next optionalreviewinterpretability/intervaltradeoffs;nonewtrainingrequested.
