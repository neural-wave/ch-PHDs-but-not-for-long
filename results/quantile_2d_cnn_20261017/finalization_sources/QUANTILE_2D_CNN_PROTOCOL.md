# Small 2D CNN with quantiles and catastrophic-day risk — 17 October 2026

User approved trying the quantile-CNN restart; approximately5.2 September MAE is
the practical target, not a test-based model-selection rule. Preserve existing
models and deployment. Official challenge data only, no external/pretrained or
October evaluation prices. Four zones predicted jointly. Exact hourly means
within0.0001EUR/MWh; recent history supplies frozen inference rules only.

Inputs: previous7calendar days of full96-quarter prices and24hourly means
repeated4times, plus target-day24known hourly prices repeated to96. Q/H paired
rows, currentQ zero/unavailable even during training; literal availability mask.
Missing calendar days keep their lag positions. Raw pack includes four price
channels, mask, current hourly prices/neighbor differences, quarter position,
weekday and four clock channels. Analog90 pack additionally exposes historical
interpolation errors/hourly differences in the grid and four fixed analogous
error curves from strictly earlier90calendar days. It is a single CNN, no blend
with an earlier trained model. The prior August-selected analog90 benchmark
completed and audited at SeptemberMAE5.177024 (separate experiment).

Fixed shared encoder: Conv2d(channels,16,2x8,stride2x1), ReLU; flatten eight
day-pair identities into channels; Conv1d(...,16,1), ReLU. Independent point4,
marginalmedian4 and quantile-increment40 output channels. Softplus cumulative
increments enforce eleven ordered marginal quantiles at
.025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975. Point residuals are mean-centered
within each hour and added to fixed mean-preserving interpolation. The q50 head
is independent; marginal quantile curves do not preserve hourly means.

Matched24fits:2inputpacks x2quantileweights(0/1) x2LR(.001/.003) x3seeds17/43/71.
All models instantiate identical heads; the MAE-only control does not train or
report its quantile head. Train302days through July. August1–15 alone selects
epoch, seed and model by pointMAE. Reserve August16–31 exclusively for calibration.
Maximum600epochs/patience60, AdamW.01, batch32, gradientclip5. Identical data
normalization, target units20EUR/MWh and disabledTF32 throughout. Save choices
before September inference. Report all four input/loss groups; no adaptive
September search, goal-driven stopping, deployment refit or submission.

Quantile evaluation: per-level pinball and empirical coverage, crossing rate,
raw and CQR-widened50/80/90/95% intervals, width, per-zone and whole-day empirical
coverage. CQR calibrates zone/quarter-position groups over16days x24hours, with
finite-sample corrected ranks. Temporal dependence invalidates an automatic
exchangeability guarantee; pointwise coverage does not imply daily coverage.

Risk event fixed before launch: joint four-zone dayMAE>10EUR/MWh. Generate182
out-of-sample earlier-day errors using6expanding folds with train/stopping-end
indices120/151/181/212/242/272 and next blocks through302. Each fold trains only
before its15-day stopping block; normalizers are fold-training-only. Architecture
hyperparameters come from August1–15 selected quantile model; this is not nested
historical model-selection evaluation. Risk uses no in-sample residual labels.

Fit four small regularized logistic probes on these labels: inputs; inputs plus
quantile/point summaries; plus conv activation summaries; plus fusion summaries.
Layer summaries aggregate mean/std/max/active fraction, avoiding filter-identity
alignment across separately trained folds. A regularized one-parameter intercept
adjustment and an alert threshold use August16–31 only. With fewer than3 examples
in either training class, report a constant-probability fallback. Report every
probe's Brier/AP/AUC, reliability bins, fixed-threshold alerts and catastrophe
recall among highest-risk10% of Septemberdays. The latter is a batch ranking
diagnostic, not a causal daily deployment threshold.16calibration/30testdays make
probability claims exploratory; no winning probe selected using September.

Export normalized kernels, fusion/head weights, signed/positive Grad-CAM and
local activation-gradient maps for point corrections and90% interval width.
Examples: first Septemberday, worst actual day, highest predicted riskday,
explicitly posthoc diagnostics. Never explain the conservation-fixed day sum.
Report empirical weight-randomization map correlations; no invented guarantee.
Day-removal ablations rebuild masks/analog retrieval consistently.

Required gates: official hash/array equivalence; input perturbation/causality;
query zero/mask and known hourly rows; finite gradients; cross-zone influence;
independent ordered q50; point conservation; short GPU fits; all24fits; August
selection; independent checkpoint/calibration/risk replay; held-out risk labels;
finite explanation artifacts. Research scores below target remain valid results.

Execution: existing VastRTX5090 root@93.91.156.85:55340,
/workspace/HACKATON,/venv/main/bin/python. Current Nightwatch command wrapper on
execution host; success only after run_quantile_cnn.py completes final audit.
Output: results/quantile_2d_cnn_20261017. Source: quantile_cnn.py,
analyze_quantile_cnn.py,audit_quantile_cnn.py,run_quantile_cnn.py.

Post-training diagnostic: the fixed10EUR/MWh event produced only1/182earlier
held-out events and0/30September events, so the predeclared logistic probes
correctly fall back to constant probabilities. This cannot validate catastrophe
detection. Add explicitly exploratory continuous daily-MAE ridge probes using
the same four feature sets,182held-out labels, fixed ridge penalty10, and a
regularized intercept adjustment on August16–31. Report every probe and the
constant-calibration baseline, Spearman/Pearson associations, and raw90% width
association with dailyMAE. Do not redefine catastrophe or select on September.
Source: analyze_cnn_confidence.py. Independent coefficient/prediction replay
is included in the final audit. Reconstruction choices/checkpoints unchanged.
