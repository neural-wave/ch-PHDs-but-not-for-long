---
title: One-layer linear S4D frequency study launched with exact interpretability
status: historical
kind: experiment
recorded_at: '2026-10-17T15:16:32.123780+00:00'
sources:
- campaigns/linear_s4d_20261017/PROTOCOL.md
- campaigns/linear_s4d_20261017/train.py
- campaigns/linear_s4d_20261017/interpret.py
- campaigns/linear_s4d_20261017/audit.py
- campaigns/linear_s4d_20261017/run.py
- results/linear_s4d_20261017/launch.json
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T16:03:16.349618+00:00'
---

User asked for very simple S4, just a layer, to make it interpretable and learn frequencies. Implemented one-layer S4D(diagonalS4), explicitly distinguished from full originalS4. Point path exactly affine in permitted raw price inputs conditional on masks: one10channel S4D core shared forward/reverse over768 quarter tokens(seven historical days plus current96 known-hourly repeats), linear4zone readout, hourly zero-mean projection, conserved interpolation anchor. No nonlinear engineered features, encoder projection, activation, LayerNorm, clipping or gate in point path. Ten named channels:4hourly,4signed past interpolation errors,known-hourly andobserved-quarter masks. Current quarterly errors remain zero/unavailable. Missing calendar blocks preserved.

Eight/sixteen real-equivalent states per input channel(4/8complex modes); total1104/1264parameters including separate ordered-quantile gap head. Samplingdelta fixed.25hour to avoid A/delta confounding. Declared initial periods1/6/12/24hours or.5/1/2/4/6/12/24/168hours, decaytimes2xperiod. Frequencies/decays/amplitudes train; matched frozen-frequency control still learns decays/amplitudes/readout. This isolates frequency-learning contribution. Nonlinear softplus uncertainty head does not change affine point path. Modes can trade amplitudes and alias; periods derived from discrete poleangle, decay from radius. Do not claim uniquely identified physical cycles. Inspect effective kernels and fold consistency.

Unchanged VALIDATION_PROTOCOL.md: eight expanding February–Septemberfolds, first15selection,16–endcalibration, fitting-only fromscratchnormalizers/models, frozen testparameters and prior-only observed quarters.16candidates/fold(two families x two statesizes x twoLR x two seeds),128fullfits. MedianMAE+eleven-quantilepinball training,300epoch/patience35. Required same monthly ratio/pooledMAE,quantile andCQR50/80/90/95reporting; no deployment change.

Linearity smoke passed: affine superposition error5.55e-17learned/7.63e-17fixed, direct recurrence error6.33e-15/9.99e-15. First launch9eb2c99f0de337971afeca9c crashed during independent smokeaudit only due pandas rejectingnumpy.str_; no fulltraining launched. Fixed pd.Timestamp(str(date)) in source/audit, preserved remote failedoutputs results/linear_s4d_20261017_smoke_failed_9eb2c99f. New outputroot fresh; no checkpointreuse.

Corrected Nightwatch d85b1d1b1b58a45af40c3a57 running on existing root@93.91.156.85:55340 RTX5090,/workspace/HACKATON,/venv/main/bin/python. Worker11206;log/root/.local/share/nightwatch-hackaton/d85b1d1b1b58a45af40c3a57.log.15-minute estimatedduration,heartbeat_expectedtrue/timeout180seconds; subsequentheartbeatverified. Repairedfirst/lastfold smoke independently passed2folds/4checkpoints,chronology,preprocessing,selection/replay/calibration/scores. Last live check fullFebruary fits progressing, no finalscoreclaimed. Current wrapper sends actualsuccess/failurealerts. Success requires full128checkpointaudit, exactsourcechannelcontribution and independentnumpyconvolution replay for all16family/foldwinners, then recoveryarchive. Outputresults/linear_s4d_20261017 withfrequencies.csv,frequencies.png,perfold interpretations/effectivekernels andcontributioncurves. Next on completion retrieve/audit/back up and inspect usefulfrequencyamplitudes/foldstability, not only training-frequencyvalues. Prior nonlinear160fit comparison remainscompleted/audited/backedup, ratiosTCN.778942,S4D.785554,S5.788695.
