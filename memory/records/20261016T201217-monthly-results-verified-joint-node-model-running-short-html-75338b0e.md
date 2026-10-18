---
title: Monthly results verified, joint-node model running, short HTML delivered
status: current
kind: experiment
recorded_at: '2026-10-16T20:12:17.093871+00:00'
sources:
- results/experiments_summary.html
- write_experiments_html.py
- results/monthly_base_gpu1/metrics.json
- results/monthly_cat_gpu1/metrics.json
- graph_models.py
- train_graph.py
- run_graph_after.py
- verify_graph.py
- results/graph_launch.json
- MONTHLY_PROTOCOL.md
experiment_id: null
supersedes:
- memory/records/20261016T200043-user-requested-august-validation-september-test-implemented--e1484989.md
---

Previous goal turn made progress implementing/launching requested302/31/30monthly split. This turn verified monthly Nightwatch33ca46c2ab0f4ebb14e99ea2 completed exit0 with bothfull-model exportedCSV checks passing. Retrieved metrics: baseAugustMAE4.2554961098/September5.1258003184; CatBoost4.2545661914/5.1227931936; corrected interpolationAugust6.2380290239/September7.0903212348. Both model_fit_days302 and final_refit_deferredtrue. Validation selection onlyAugust; September not fitted, though seen in earlier development. Local metrics under results/monthly_base_gpu1 and monthly_cat_gpu1. Current monthly artifacts still remote pending selected-winner retrieval.

Implemented joint four-node graph/temporal comparison to pursue error minimization after user's question about joint inputs. GraphNet takes day×4nodes×24hours×389features, shared temporal convolutions, learned fully-connected four-node multiheadattention, LayerNorm/FFN residual blocks, four-quarter zero-mean outputs. Random initialization onlyofficialdata. Sixteen width/block/dropout/lr/seed variants; gradients on302days, early stopping and convexmixtures onAugust31days, September onlyafterselection. Parent monthly ensemble is checked by sourcehash, exactcachepredictions/validationMAE, splitcounts302/31/30, 302-day fit and successfulQA. No production all-data refit. Standard predict.py supports graphstate inference alongside nestedparent. graph-only exportedCSVQA ensures graph path is exercised even if smokeensemble selects onlyparent.

Remote verify_graph.py passed meaningful CPU checks: actualcross-node influence, node-permutationequivariance, independentdaybatches, expectedoutputshape, finitezero-sum residuals. Remote compilationpassed. Staged /home/mnogales/Projects/HACKATON/graph_campaign. Queued thenverifiedrunningNightwatch a08c4ed9d064856b2beaf5bd, supervisor2157738; previousmonthlycompletedbeforeGPUtraining. Heartbeatexpected180s and Telegramcompletion/failurealerts enabled. Log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/a08c4ed9d064856b2beaf5bd.log; outputs results/monthly_graph_gpu1. InitialETA25min(~22:32Zurich). Latestlog reachedgraph_8 andfullsearch; graphsmoke thereforepassedincludinggraph-onlyCSVQA. Finalmanifest comparesbase/CatBoost/graph byAugustonly.

User additionally requested a shortHTML on things tried. Created results/experiments_summary.html (~4.9KB standaloneCSS/noexternalassets) and reusable write_experiments_html.py. It reports verifiedcurrentmonthlycards againstsame-periodinterpolation, separates earlier60-daytesttable, describesricherfeatures/zoneweights/boosting/correctionattempts, honestlyrecords supersededrolling-smokefailure, andmarksjoint-node results pending. Generator choosescompletedlocalreportsbyAugustonly. HTMLParser checkpassed; allevidence-relative linksresolve. Open-in-Codex wasqueued forcurrentchat. AfterretrievingcompletedmonthlyCatBoostmetricsregeneratedHTMLwithcurrentbestnumbers. SnapshotmetadataEurope/Zurich. No pendingaccuracyclaimforgraph. Nextauditgraphterminalresult/eligibility/302-dayfit, retrievehash-verifiedwinner, refreshHTML. Goal remainsactive.
