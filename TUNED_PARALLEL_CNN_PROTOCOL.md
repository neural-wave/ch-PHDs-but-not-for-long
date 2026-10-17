# Focused improvements toward MAE5.1–5.2 — 17 October 2026

User asked how to improve the small parallel2D model from September5.431127,
ideally reaching5.1–5.2. This is a target, not an established attainable score.
Preserve the confirmed Q/H parallel rows, seven-day CNN grid,2x8 kernels,
jointfour-zone continuous outputs, simplicity and exact hourly conservation.

Seven arms, twoLRs(.001/.003), three seeds17/43/71 =42fits:
control(original8filters),width16,clock(width16+hour/yearsin/cos),
residual(clock+fourextra grid channels),analog7(residual+4late analogcurves),
analog90(larger strictlyprior matchingwindow),context(analog90+fusionwidth5time).
All remain one2D encoder + one small fusion layer + linear output; no U-Net,
attention, learned ensemble or existing-model blend.

Raw quarter/hour rows and availability masks are preserved in every arm. The
residual arm adds zone channels whose quarter row is prior actual price minus
mean-corrected interpolation (/20), hourly row is prior hourly minus queryhourly
(/100). Current-day residual rows are zero; current quarter targets always
masked. This exposes historically unexplained quarter movements separately from
pricelevel. Clock adds sin/cos of time-of-day and year. Analog curves use
up to10 nearest prior localhourly shapes in7 or90calendar days, fixed weighting
1/(25+distance), distance=local5hour shapeMSE+.1leveldifference². These are
weighted prior interpolationerrors, added as4late channels. No target-based
matching, inference-time fitting or current/future quarter observations.
The90day arm expands only analog retrieval; CNN grid remains7days. No causal
claim is inferred from earlier frozen-model featureimportance findings.

Train302/August31/September30development, train-only normalization, AdamW.01,
MAE,batch32,gradientclip5. Max600epochs/patience60 for allarms; control rerun
isolates broader optimizer/stopping budget from architectural changes. Original
control feature/model parity must pass. Select each arm and overall model using
August only; save choices before September inference. Report all predeclared
arms, including failures. Never continue adaptive September score hunting or
choose the smallest September score as winner. No deployment/refit/submission.

Smoke: original control parity, futuretarget perturbation and past-only
equivalence, missing masks/residuals, cross-zonegradient/conservation,7arm
shorttraining. Independent audit reconstructs feature/statistics and predictions,
officialinputequivalence, all42fits, Augustchoices, metrics/conservation and
serializedreplay. Selected kernels export normalized-input weights for raw and
residual zone0 channels; no intrinsic causal attribution claim.

Sources:tune_parallel_cnn.py,audit_tuned_parallel_cnn.py,run_tuned_parallel_cnn.py.
Outputs:results/parallel_2d_tuning_20261017. Existing VastRTX5090,
root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python; four workers,
Nightwatch durable heartbeat/success/crash wrapper. Exact counts and actual
results saved in trial/model.pt and metrics/report/audit artifacts.
