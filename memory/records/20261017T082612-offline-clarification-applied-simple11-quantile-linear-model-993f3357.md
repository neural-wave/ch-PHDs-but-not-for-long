---
title: 'Offline clarification applied: simple11 quantile linear model with hierarchical
  median launched'
status: historical
kind: experiment
recorded_at: '2026-10-17T08:26:12.338181+00:00'
sources:
- LINEAR_QUANTILES_PROTOCOL.md
- linear_quantile_model.py
- train_linear_quantiles.py
- plot_linear_quantiles.py
- verify_linear_quantiles.py
- run_linear_quantiles.py
- results/expanding_vast/metrics.json
- results/expanding_vast/learning_curves.csv
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T08:38:37.751257+00:00'
---

2026-10-17 user clarified “online” meant keep going, not online/sequential parameter updates. Ongoing authorized scope:11 quantiles,joint objective,medianMAE,hierarchy,conformal,expanding/fixed-test curves,simple/interpretable model. No online updating/deployment implied. Previous neural11quantile discussion had not yet launched any quantile training; this turn implements a simpler linear model after user's simplicity preference.

Implemented one shared linear layer389namedrichcausalfeatures→43outputs:three mediancontrasts(a,u,v),40nonmedianquarteroutputs(10quantiles×4quarters). Median correction(a+u,a−u,−a+v,−a−v) added to deterministic mean-correctedinterpolation. Readable rawcoefficient contributions andplainmodelJSON;notaneuralencoder/attention/ensemble.11levels .025,.05,.1,.2,.25,.5,.75,.8,.9,.95,.975. Jointloss=medianMAE+meanpinball11+L2, equalunitMAE/pinballweights. Fourconfigs lr.001/.003×L2.001/.01,seed17,max250epochs/patience25,batch2048,remoteGPUfloat64. Train-only quantilebiasinit andnormalization;selectepochs/configbyunregularizedjointvalidationloss. Medianpointconserveshourlymean. Rawnonmedianquantilescan cross:explicitclamp/sortaroundfixedmedian;report rawcrossing andraw/postprocessed11pinball/coverage. Coefficients explain rawfunctions;sorting switchesactivefunctions. Do not enforcehourlymeanoneverymarginalquantile;wouldcollapse ordereduncertainty.

Same16folds aspriorstudy:8expandingOct–Dec onward/nextmonthval/followingtest,8fixedtrainendJuly/valAug/testSept with3–10monthshistory. Val1–15select,16–endcalibrate;freshfit,notestselection,noonlineupdates. Compare rawquantile bounds,median-residualconformal andwiden-onlyCQR (nonnegativecalibrationscores),50/90/95intervalsummaries;saveall11predictions. Additional30minute middle-stage residualconformal frommedianpairaverages;never sumquarterquantilestoasserthalf-hourquantiles. Temporalcoverageempirical. Resultsremote /workspace/HACKATON/results/linear_quantiles_vast;readableper-foldmodel.json/coefficients.csv/testNPZ/metrics;learning_curves.csv,interval_curves.csv,medianMAE/pinball/coverage-width/all11calibrationPNG/SVG.

Localsyntaxchecks passed;remoteGPUpreflight passes11levels,jointloss/finitegradients,hierarchicalmean,ordering,independentNumPycoefficientreplay. Initial trackedbdb5ef7e86a517aa990c8e2e crashedinsmokebeforefullrun because testmonth/testmetrics keycollision;renamed test_metrics. Foregroundcorrected2-fold2-epochsmoke completed allmodel/interval/replay/plots. Scoresaresmokeonly. RestartNightwatch0bf41a85e0867de319a9e14a verifiedrunning,heartbeatexpectedtrue180s,worker2511. Log /root/.local/share/nightwatch-hackaton/0bf41a85e0867de319a9e14a.log. Started10:24:59ZurichOctober17,ETA10m≈10:35;actualterminalstate required. Detachedwrapperrerunspreflight/smoke/fullandplots. Nextverifystatus/retrieve/audit/visuallyinspectquantileresults;noquantileaccuracyclaimyet.

PriorMAE-only16folds completed allfolds/plots/audits. Retrievedoverall/per-zone/hierarchybenefit/stagePNGs plusmetrics/CSVunderlocalresults/expanding_vast. VisuallyinspectedfixedSeptemberlearningcurve:readable. ExpandingweightedMAE ridge4.3732050691,hierarchy4.4197185061,direct4.4332154709,interpolation5.8148171208. FixedSept10monthfold hierarchy5.2547073306,direct5.3425531594,ridge5.2942215695. Fixedhistorycurvefrom3→10monthshierarchy≈5.72→5.25,nonmonotonic. This reference isMAE-only,notnewquantileresult. Existingmonthlyanddeploymentmanifests preserved.
