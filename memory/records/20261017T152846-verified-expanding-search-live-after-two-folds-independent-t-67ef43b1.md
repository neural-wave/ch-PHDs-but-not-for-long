---
title: Verified expanding search live after two folds; independent training-size comparison
  prepared
status: historical
kind: progress
recorded_at: '2026-10-17T15:28:46.192933+00:00'
sources:
- campaigns/minimize_expanding_20261017/summarize_verified.py
- results/minimize_expanding_20261017/launch.json
- results/minimize_expanding_20261017/smoke_audit.json
experiment_id: minimize_expanding_20261017
supersedes: []
status_updated_at: '2026-10-17T15:59:43.434775+00:00'
---

Goal continuation 17October2026: previous goal turn classified as progress (implemented and launched384fit search with independently audited preflight). Rechecked authoritative execution state rather than restarting. Nightwatch0c174584a3b9236d9f21ffd4 stillrunning, heartbeat1792250829.37209, exitcodenull, heartbeat_expectedtrue,180secondtimeout,Telegramenabled. Execution-host actual PIDs11868 (/venv/main/bin/python campaigns/minimize_expanding_20261017/run.py0) and12025 (/venv/main/bin/python .../train.py run) observed live. progress.json showed2/8completedfolds,lasttest2026-03; log confirms48/48Marchfits and EXPANDING FOLD DONE2/8. Hostroot@93.91.156.85:55340,/workspace/HACKATON;log/root/.local/share/nightwatch-hackaton/0c174584a3b9236d9f21ffd4.log. No terminalresult or aggregatequalityclaim.

Independent useful preparation completed: campaigns/minimize_expanding_20261017/summarize_verified.py created,PythonASTchecked anduploadedtoexecutionhost. It is reporting-only and doesnotchange the manifest-frozen training/selection/audit code. It refuses to run without fullcompletion/audit384candidate scores/eightfolds. Revalidates sourcehashes; independently recomputes interpolation, old/newselector pointMAE andratios from matched savedquarterarrays; checks dates/truth/q50conservation/quantileorder and intervalnesting; verifiesincreasingtrainingmonths3through10and241testdays. Produces selector_comparison_by_training_size.csv, selector_interval_comparison.csv,verified_selector_comparison.json and standalone expanding_selector_comparison.png/.svg. Shows month-by-monthdifferences,trainingcounts,pooledMAE,monthlymeanratio,worstmonth and50/80/90/95coverage/width. Statestrainingdurationandcalendarmonthco-vary,so not an isolatedtrainingsizeablation. Requires actualcompletion,notelapsedETA; sourceaddedwhileworkrunning,notexecutedyet.

Next: on confirmedcompletion execute /venv/main/bin/python campaigns/minimize_expanding_20261017/summarize_verified.py ontheVastGPUhost, inspectfullaudit/report/selectorcomparison, retrieveandSHA256verifyrecoveryarchive, renderinspectplot andreport measuredresults. Keepgoalactivebecausefulloptimizationresultsremainpending. Estimated17:45Zurich remainsonlyestimate; tracker handlesactualsuccess/crashalerts.
