---
title: Single-layer interpretable S4D completed audited and locally backed up
status: current
kind: experiment
recorded_at: '2026-10-17T16:03:16.355431+00:00'
sources:
- results/linear_s4d_20261017/REPORT.md
- results/linear_s4d_20261017/INTERPRETATION.md
- results/linear_s4d_20261017/metrics.json
- results/linear_s4d_20261017/frequency_summary.json
- results/linear_s4d_20261017/frequencies.csv
- results/linear_s4d_20261017/interpretability_audit.json
- results/linear_s4d_20261017/backup_verification.json
experiment_id: null
supersedes:
- memory/records/20261017T151632-one-layer-linear-s4d-frequency-study-launched-with-exact-int-bc276987.md
---

Verified2026-10-17: single-layer S4D run d85b1d1b1b58a45af40c3a57 completedexit0 at15:24:39UTC(17:24Zurich). All128fits across eight expanding February–September folds passed independent chronology/train-only preprocessing/checkpoint/selection/calibration/score audit. Required first15day selection and16–end calibration retained. Conservedq50,eleven ordered quantiles,calibrated50/80/90/95intervals saved;92544quarters. Full originalS4 not tested: explicitly S4D diagonal variant. Point path exactly affine conditional on masks; quantile-gap head separately nonlinear.

Learned-frequency primary mean monthly ratio.8104977013,pooledMAE4.6980305192EUR/MWh,calibrated90%coverage.9078708506,width24.3086071EUR. Frozen-frequency control.8177956273,4.7417224379,.9126901798,width25.6991675. Both still learn decays/amplitudes/readout; onlyfrequencies frozen. Relative ratio improvement.89239%,or18.9502%reduction versus interpolation. Pre-test validationselector chose learnedfamily in allfolds and matchedsameaggregate. Previous nonlinearTCN ratio.778942 was stronger; simplemodel tradesaccuracy forcompact exactdecomposability. Developmental historical backtests, no significance or blind-test claim, no deployment change.

All16family/foldwinners passed exactsourcechanneldecomposition and independent NumPy convolution replay. Maximum errors1.59872e-13 and3.37508e-13respectively. Exported per-fold firsttestday baseline/point/inputcontributions, lagkernels/readout/feedthrough/normalizers andeffectivekernels forhours0/12/23. Point mapping explainsmodel exactly, not causalmarketmechanisms.

Frequency inspection: median absolute periodshift3.22158% from declared initialanchors, unweighted across selectedlearnedmodes/channels/folds. Initial12hband median11.9145h,24hband23.4669h; these anchors were supplied and not newly discovered.168h initialized band median178.709h,range98.0655–282807.251h; near-zero digitalangles become nearlynonoscillatory overwindow, not observedmulti-year cycles. Negative realpoleparts stillstable. Seven-day history limits weekly/long-periodidentification. Markeramplitude is readoutcoefficientproxy, not physical spectralpower or causalimportance, and hourlymeanprojection can cancel slowcomponents. Inspect effectivekernels/foldconsistency. Summaryfrequencyanalysis uses tiny localCSV only; alltrainingremoteVast.

Hostroot@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python,RTX5090;log/root/.local/share/nightwatch-hackaton/d85b1d1b1b58a45af40c3a57.log. Remote/localarchive results/linear_s4d_20261017_completed.tar.gz333149705bytes;SHA256f1910a8204d197012990c76a5edab76e080afeac12fded998ea016791efd59e8 matchesreceipt. Allresults/checkpoints locallyextracted,backup_verification.json true;noactiveSSMtraining/download. INTERPRETATION.md andfrequency_summary.json writtenlocally afterrecovery. Otherprojectcampaigns not interrupted. Originalfailedsmoke9eb2c99f remains preserved from priorrecord. Next optionalinspect exactkernel/contributionexamples or comparemoreinterpretable models; do not launchnewfit basedonlyonstatusrequest.
