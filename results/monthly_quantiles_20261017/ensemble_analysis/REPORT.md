# Ensemble weights and model specialization

Posthoc analysis of completed quantile predictions; no router was fitted and no forecast was changed. Current weights are fixed for a whole test month and selected using the preceding first15validationdays. Quantile functions are averaged at every level, then blend intervals are calibrated separately.

## Composition of the best aggregate recipe (ordinary training)

| Test month | Components |
|---|---|
| 2026-02 | Additive 33.3%; MLP 33.3%; Temporal CNN 33.3% |
| 2026-03 | Rich linear 50.0%; MLP 50.0% |
| 2026-04 | MLP 33.3%; Temporal CNN 33.3%; Analog 2D CNN 33.3% |
| 2026-05 | Rich linear 33.3%; MLP 33.3%; Temporal CNN 33.3% |
| 2026-06 | Compact linear 33.3%; Temporal CNN 33.3%; Analog 2D CNN 33.3% |
| 2026-07 | MLP 33.3%; Temporal CNN 33.3%; Analog 2D CNN 33.3% |
| 2026-08 | Rich linear 33.3%; Temporal CNN 33.3%; Analog 2D CNN 33.3% |
| 2026-09 | Rich linear 33.3%; MLP 33.3%; Analog 2D CNN 33.3% |

## Conditional scores

Within each context, the ratio divides model MAE by interpolation MAE on that same subset and averages monthly ratios. Counts are whole zone/hours, each containing four quarters. These conditional scores are distinct from the overall eight-month metric. Curves use fitting-only slope thresholds. All these comparisons remain developmental and posthoc.

| Context | Best single expert (ordinary training) | Ratio | Blend ratio | Months won by that expert |
|---|---|---:|---:|---:|
| Zone: ZONE_0 | Compact linear | 0.615514 | 0.607889 | 4/8 |
| Zone: ZONE_1 | Rich linear | 0.668846 | 0.664617 | 5/8 |
| Zone: ZONE_2 | Analog 2D CNN | 0.829255 | 0.836246 | 7/8 |
| Zone: ZONE_3 | Compact linear | 0.796699 | 0.791679 | 4/8 |
| Curve: Falling | Compact linear | 0.683540 | 0.675405 | 2/8 |
| Curve: Flat | Compact linear | 0.982636 | 0.970097 | 5/8 |
| Curve: Peak | Rich linear | 0.789886 | 0.782300 | 0/8 |
| Curve: Rising | Compact linear | 0.694947 | 0.685053 | 1/8 |
| Curve: Trough | Analog 2D CNN | 0.911111 | 0.898267 | 3/8 |
| Uncertainty: Middle | Compact linear | 0.782679 | 0.768828 | 1/8 |
| Uncertainty: Narrow | MLP | 0.927979 | 0.888223 | 2/8 |
| Uncertainty: Wide | Compact linear | 0.666339 | 0.665213 | 1/8 |

## Complementarity and routing bounds

{
  "uniform": {
    "mean_blend_ratio": 0.7204941118854624,
    "hindsight_oracle_whole_hour_ratio": 0.5569259581094569,
    "mean_weighted_member_ratio": 0.7474992765170948,
    "mean_cancellation_gain": 0.027005164631632363,
    "blend_beats_every_expert_hours_percent": 5.419666858678955
  },
  "ratio": {
    "mean_blend_ratio": 0.7216795826329767,
    "hindsight_oracle_whole_hour_ratio": 0.5605275747370294,
    "mean_weighted_member_ratio": 0.7474357938664877,
    "mean_cancellation_gain": 0.02575621123351105,
    "blend_beats_every_expert_hours_percent": 5.523073476702509
  }
}

The weighted-member ratio is the error if averaging constituent absolute errors rather than predictions. Its reduction after prediction averaging quantifies cancellation, not a router. The hindsight oracle chooses the expert with lowest true four-quarter error for each whole zone/hour, preserving the hourly mean. It reads the answer and is only an optimistic diagnostic bound. It is not a measured attainable routing gain. Some hours are better served by the blend than every expert.

An actual interpretable gate could use a small tree or additive gate from known current hourly slopes/price/spread, prior interpolation residuals and expert uncertainty/disagreement. It must be trained using earlier out-of-sample expert errors, selected chronologically, and tested on the next month. Route or mix all four quarters of each hour consistently and calibrate the final gated quantiles separately. No such gate was trained in this analysis.

Files: weights.csv, conditional_monthly_scores.csv, conditional_summary.csv, complementarity.csv, signed_error_correlations.json, explorer.json and audit.json.
