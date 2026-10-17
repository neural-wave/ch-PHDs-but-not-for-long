---
title: 2D CNN quantile restart and catastrophic day risk design 20261017
status: historical
kind: design
recorded_at: '2026-10-17T11:47:09.916936+00:00'
sources:
- README.txt
- PARALLEL_2D_CNN_PROTOCOL.md
- parallel_zone_cnn.py
- LINEAR_QUANTILES_PROTOCOL.md
- linear_quantile_model.py
- TUNED_PARALLEL_CNN_PROTOCOL.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T12:05:10.428887+00:00'
---

# 2D CNN quantile reconstruction and day-error risk: design discussion, 17 October 2026

## User request and clarified input
The user asked to restart the design around direct quantile training, MAE evaluation with exact hourly mean conservation, a simple interpretable 2D CNN, paired N-day hourly/quarter-hour histories, Grad-CAM, and predicting catastrophic-error days from available inputs. The user explicitly clarified that the additional 24 points are the target day's 24 known hourly prices. This is a design assessment; this turn neither trains a new model nor cancels previous jobs.

## Verified sources
README.txt says official inference inputs are 24 hourly prices per zone for the query date and quarter-hour observations from strictly earlier dates. Four anonymized zones, 363 valid dates, DST-change dates omitted; no external data/models or pre-freeze October 1–18 prices. Recent history can feed frozen inference rules, but does not authorize inference-time retraining or recalibration.
parallel_zone_cnn.py / PARALLEL_2D_CNN_PROTOCOL.md already implement paired Q/H rows, repeated hourly values, a zero query Q row, explicit masks, and a small joint-zone 2D encoder; these are precedents, not evidence that quantile CNNs or risk detection work.
LINEAR_QUANTILES_PROTOCOL.md / linear_quantile_model.py already distinguish mean-conserving point forecasts from marginal quantiles. Their prior quantile treatment constrains the central decision and rearranges bounds; a fresh statistical design can instead keep a separate unconstrained marginal-quantile head.

## Proposed minimal restart (recommendations, not user-approved experimental settings)
For each zone, pair Q and H rows for N prior calendar days and the query day. Repeat each H value four times. Query-day H is supplied; query-day Q is zero and unavailable during training and inference. Add an explicit availability mask because zero prices are valid. Preserve missing calendar positions, quarter-within-hour position and weekday. A reasonable first N is seven; this remains a proposal.
Input layout can be batch x (four zone channels + mask) x 2(N+1) rows x 96 quarters. A shallow 2x8 Conv2d with stride two over rows gives interpretable paired Q/H feature maps by lag and time; a small fusion layer retains lag identities and combines all days/zones. These kernels span the two representations, not neighboring calendar days; a day-by-time convolution is a separate architectural choice. Eight to sixteen filters are sufficient as an initial budget, not a performance claim. Use linear physical-price outputs, allowing negative prices.

Two heads share the encoder:
1. An MAE decision head emits four within-hour residuals and centers them: prediction[h,q] = H[h] + r[h,q] - mean_q(r[h,q]). A fixed mean-preserving interpolation anchor may replace H repetition in a matched comparison. Compute MAE after this operation inside training.
2. An independent ordered marginal-quantile head predicts quarter-hour quantiles using pinball loss, for example the existing eleven levels. Ordered increments can prevent crossings. Do not force its q50 to equal the constrained decision if the aim is genuine marginal quantiles: the feasible MAE-optimal vector need not equal coordinatewise conditional medians.
Joint objective: constrained point MAE + lambda * mean quantile pinball. Select by point MAE while reporting pinball and calibration; additional quantile training need not improve point MAE, so use a matched MAE-only control.

For ordered quantile curves L <= U, imposing sum_q L = sum_q U = 4H implies sum_q(U-L)=0 and hence L=U coordinatewise. Therefore only the point decision has the hourly conservation constraint; quantile endpoints are not jointly realizable daily scenarios. If coherent random scenarios are later needed, generate samples in the zero-sum within-hour residual subspace, enforce means on each sample, and derive marginal quantiles afterward. Quantile endpoints cannot simply be added to obtain aggregate quantiles.

## Interpretation
Inspect normalized-input kernels, activation patterns and representative activating windows. CNN size/2D weights do not establish causal or faithful interpretation. Adapt Grad-CAM to scalar targets: a quarter correction, within-hour contrast, interval width, or learned risk score. Avoid a day/hour sum as attribution target: the residual output sum is fixed, producing zero or uninformative gradients through the learned reconstruction branch. Explain the correction branch explicitly when the baseline is a skip path. Signed attributions are useful for negative contributions. Validate with model-weight randomization and consistent day/zone/representation ablations; keep Q/H derived quantities consistent when perturbing history.

## Catastrophic-day risk and calibration
Quantile width describes predicted conditional spread, not a calibrated probability that this model's MAE will be large. Predefine an operational error event, e.g. joint-zone day MAE above a fixed T EUR/MWh; peak-quarter or worst-zone events are distinct and should be reported separately.
Build chronological out-of-sample errors from earlier official training blocks. Fit a small risk estimator on available input summaries and out-of-sample interval-width summaries; optional frozen-layer probes can test whether intermediate representations help. Never use in-sample training residuals as the sole risk labels, future prices, or test-day realized error as a predictor. Keep prediction-model selection, risk fitting/calibration and final evaluation chronologically separate. Candidate signals include hourly ramps, cross-zone spreads, historical within-hour volatility and novelty relative to earlier inputs.
Measure precision-recall, probability reliability/Brier score, catastrophe recall at fixed alert rate (e.g. flag top 10% of days), and actual MAE across risk bins. Use days as the sampling unit because quarters and zones are correlated. With 363 days and rare extremes, uncertainty is substantial. Similar observable hourly curves can conceal different quarter-hour spikes, so perfect advance detection may be impossible.

Calibrate quantile intervals on held-out chronological days, separately from model/threshold selection. CQR is a candidate, but its ordinary finite-sample guarantee assumes exchangeability and does not automatically hold under temporal dependence/drift. Quarter-level coverage is not whole-day simultaneous coverage or catastrophe detection. September is already developmental; genuine confirmation requires a later untouched period. Challenge rules prohibit adapting calibration at inference after freeze.

## Primary research checked
- https://arxiv.org/html/1905.03222v1 : pinball loss and conformalized quantile regression; theorem requires exchangeable observations.
- https://arxiv.org/abs/1610.02391 : Grad-CAM applies to differentiable scalar target concepts and structured CNN outputs.
- https://arxiv.org/abs/1810.03292 : saliency sanity checks; visual appeal alone does not validate explanations.
- https://arxiv.org/abs/2106.00170 : distribution shift challenges ordinary conformal exchangeability assumptions; online adaptation is not proposed here given challenge rules.

## Next step
Present the design and tradeoffs; then specify the bounded MAE-only versus MAE-plus-quantile comparison and choose the catastrophe event before implementing/training a new campaign. Read the Nightwatch guide and verify the authorized remote resource before any long run. Earlier jobs below were not live-audited in this turn.

## Startup summary preserved verbatim before replacement
# HACKATON project memory

Objective: official-only four-zone quarter-hour reconstruction, hourly means within 0.0001 EUR/MWh. No external/pretrained/October prices; strictly earlier history. Train 302 days through July, August 31 selection, September 30 developmental evaluation, not blind. Zones anonymized.

Verified 2026-10-17: best monthly_joint_blend_gpu1 September MAE5.091209; interpolation7.090321. Offline363-day refit audited; no external submission. DEPLOYMENT_REFIT.md; results/COMPLETION_AUDIT.md. Minimal1D joint CNN history5.539996, no-history5.921824. Confirmed parallel2D Q/H kernel2x8: history5.431127, no-history5.903767;1388/940parameters, final audit passed, checkpoints/predictions/filterplots locally backed up. PARALLEL_2D_CNN_PROTOCOL.md; results/parallel_2d_zone_cnn_20261017.

Active 2026-10-17: user targets2D MAE5.1–5.2 while keeping simplicity. Focused42fits: control,16filters,clock,interpolation-error grid channels,analog7/90,temporal fusion. Most variants under5kparameters; context15124. Seven-day grid retained; analog90 retrieves older strictly prior official curves. TUNED_PARALLEL_CNN_PROTOCOL.md; results/parallel_2d_tuning_20261017. Smoke/causality/nonzero-control parity passed; full training9/42 observed, final scores/audit pending. August-only choice frozen before September.

Remote VastRTX5090 root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python. Nightwatch038fce95c7be0dda153059b8 worker5203, heartbeat180s, initialETA15min; log/root/.local/share/nightwatch-hackaton/038fce95c7be0dda153059b8.log. Next retrieve final audited results; no deployment change.

Other context preserved in memory/records/20261017_startup_before_2d_tuning.md: earlierCNN TF32 fix; segmented64fit run270e46a91311c053123bd97a unrechecked; frequency results audited; expanding/conformal034d5133ebc5e076af6aed88 audit outstanding. Analog reliance motivated study, not proof of improvement. User values interpretable filters and robustness; SSM untested.
