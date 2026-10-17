# Interpretable analog corrections and difficult-day robustness

Direction agreed with user, 2026-10-17. Design below is proposed; no new fit or improvement claimed.

## Candidate models

1. Compact historical-analog residual model. Match only prior observed days/hours using training-scaled curve shape, local ramps, level and cross-zone spreads. Retrieve interpolation-error quarter vectors from those matches. A normalized nonnegative weighted average preserves their zero hourly sums. Expose dates, matching distances, weights and residual vectors.
2. MAE-trained sparse additive residual model. Readable effects of ramp, curvature, zone spread, recent signed residuals and analog summaries. Use three zero-sum contrasts and evaluate/optimize reconstructed quarter MAE jointly. A small spline basis and regularization limit complexity; existing squared-error compact ridge benchmark is not this experiment.
3. Small rule-gated version of model1 or2. At most a few rules or shallow tree on observable novelty, analog disagreement and ramps; leaf-specific shrinkage toward a validated conservative predictor. Compare with constant shrinkage and ungated versions. No rule may depend on the target-day quarter prices.

## Confidence and conservation

Use nearest-match distance, effective number of distinct historical days, disagreement of matched residuals, and recent *observed* forecast errors. Fit any gate using chronological out-of-fold errors, never in-sample residuals or future labels. Distinguish high predicted error from evidence that the alternative is better. Confidence is a hypothesis to validate, not a calibrated probability.

Prediction is baseline + reliability * zero-sum residual correction. Reliability is shared across four quarters of an hour (or entire day); quarter-specific scaling requires reprojection. If blending two predictors, both must conserve means and weights must be shared within each hour. A weighted average of historical zero-sum residuals is coherent; coordinatewise medians generally are not. Baseline selection must be validated: interpolation is not automatically safer on unusual days.

## Difficult-day audit and selection

First quantify each model's daily errors, worst-decile share of total absolute error, error by zone/hour, and paired gains on the same difficult dates. Test whether failure associates with permitted observable features: ramps, daily range, negative-price hours, unusual cross-zone spreads, historical novelty, analog disagreement and recent error shifts. These are candidate associations, not established findings or causal explanations.

Use chronological out-of-fold predictions. Define any difficult-day cutoff/rule inside fitting/selection windows. Retrospective worst-day sets may diagnose failures but cannot route predictions or select thresholds using held-out outcomes. Report mean official MAE, worst-decile mean daily MAE, worst month and zone, and deterioration versus the ungated model on common fixed subsets. Per-model tail scores are supplemental because they can refer to different dates.

Keep average MAE primary. Tail weighting/CVaR is a small explicit tradeoff ablation only; do not silently change the challenge objective, remove bad dates, or winsorize evaluation targets. Select on a fixed chronological protocol; use paired temporal block uncertainty. All existing historical months are development evidence.

## Deliverable

One per-day explanation: baseline curve; contribution curves; matched examples and weights; confidence signals; active rule; final coherent curve. Compare a small model set, retain the simplest model with consistent gains, and document any mean-versus-tail tradeoff. Official-data-only, causal history and offline reproducibility constraints remain in force. Existing submission and campaigns are preserved.

Evidence: INTERPRETABLE_PROTOCOL.md; results/feature_importance_20261017/ANALYSIS.md; results/challenge_rules_extracted.txt. Additive model reference: https://interpret.ml/docs/ebm.html (framework reference, not proof of accuracy here).
