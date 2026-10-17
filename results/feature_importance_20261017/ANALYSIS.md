# Feature importance analysis — 2026-10-17

Frozen August-selected linear, additive, MLP and CNN models from
`results/simple_5090/selection.json` were evaluated on August and September.
This analysis does not cover the selected monthly blend or all-data submission.
No parameters were changed. CPU analysis ran on Ando under
`/tmp/hackaton_feature_importance_20261017`; results are preserved locally.

## Main findings

Values below are the increase in September MAE (EUR/MWh) after grouped
whole-day shuffling, averaged over 20 permutations. Larger values mean stronger
reliance. Group effects overlap, interact, and must not be added.

| Feature group | Linear | Additive | MLP | CNN |
|---|---:|---:|---:|---:|
| Current daily hourly curves | 0.051 | 0.045 | 0.210 | 1.094 |
| Local hourly curves | 0.032 | 0.104 | 0.163 | 0.187 |
| Slopes, curvature and magnitudes | 0.143 | 0.226 | 0.256 | 0.122 |
| All historical interpolation errors | 0.462 | 0.386 | 0.105 | 0.061 |
| Errors from similar historical curves | 0.370 | 0.284 | 0.075 | 0.037 |
| Exact interpolation-error lags 1/2/7 together | 0.030 | 0.048 | 0.000 | 0.005 |
| Calendar | -0.002 | 0.000 | 0.007 | -0.004 |

Historical interpolation errors are the leading tested group for linear and
additive models (8.87% and 7.34% MAE increases). Their largest individual
component is the fixed nearest-curve analog: ten closest local hourly curves
within the previous 90 calendar days, weighted by curve/level similarity.
This suggests that matching historical conditions is more useful to these
frozen models than assuming the same pattern repeats exactly 2 or 7 days later.

For the MLP, slopes/curvature and hourly curves dominate. For the CNN, current
daily curves dominate (20.95% MAE increase), followed by local curves and slopes.
Interpolation history still has a positive aggregate effect for both networks.
The aggregate-history and analog effects are also positive on August, supporting
the direction across these two months; August was used for selection.

## Which historical days?

| Exact error lag | Linear | Additive | MLP | CNN |
|---|---:|---:|---:|---:|
| Yesterday | 0.0316 | 0.0461 | 0.0104 | 0.0053 |
| Two days ago | -0.0030 | 0.0033 | -0.0059 | -0.0004 |
| Seven days ago | -0.0010 | -0.0034 | -0.0046 | -0.0021 |

Yesterday has the clearest direct-lag signal for linear/additive models.
The exact 2-day and 7-day lags have little positive marginal reliance in this
September evaluation. Negative values mean shuffling slightly improves error;
they do not establish that a feature should be removed after retraining.

Descriptive 95% circular seven-day block bootstrap intervals for the combined
lag1/2/7 effect are [0.0117,0.0506] linear, [0.0339,0.0632] additive,
[-0.0111,0.0124] MLP and [-0.0113,0.0176] CNN. These use 2000 resamples
of paired daily errors and do not account for model-selection uncertainty.

## Interpretation and limitations

- Correlated hourly inputs, slopes and history summaries can substitute for
  one another. Separate group permutation can also create unrealistic input
  combinations. These are model-reliance measurements, not causal effects or
  expected gains from adding/removing features and retraining.
- Whole-day permutation preserves all zones/hours and quarters together. It
  cannot assess zone identity or hour identity because those inputs are constant
  across days. Their zero permutation values in JSON are uninformative.
- Calendar results concern within-month date variation. Month/season effects
  cannot be assessed fully in this design.
- Prediction anchors are held fixed. Linear/additive/MLP use interpolation;
  CNN uses flat hourly repetition. Therefore their hourly-curve reliance is not
  directly comparable as a measure of total dependence on hourly prices.
- September was previously inspected; this is development evidence. The 30-day
  sample and two evaluated months limit generalization. Intervals are descriptive.
- A second perturbation replaces standardized group values with zero. It agrees
  that aggregate interpolation history matters, but individual signs differ,
  illustrating sensitivity to the intervention. Zero here means training mean,
  not a literal zero historical error.

Recommendation: retain aggregate/analog interpolation history and slope inputs.
The new ordered seven-day input is plausible but unproven: existing exact weekly
lags are weak, so seven consecutive days must earn their place through matched
retraining. A targeted next study should separately compare signed sequences,
absolute-error/volatility summaries and analog history, using the same splits,
model settings and seeds. No new feature training was performed here.

## Evidence and verification

- Script: `analyze_feature_importance.py` in the project root.
- Full groups, August/September rankings, paired daily effects, permutation
  variability and block intervals: `output/importance.json`.
- Chart: `output/importance.png` and `output/importance.svg`.
- Model checkpoints: `results/simple_5090/trials/{168,173,190,162}/model.pt`.
- All four baseline MAEs reproduce the recorded scores within 2e-5 EUR/MWh;
  CPU predictions reproduce cached predictions within 1e-4 EUR/MWh.
- The ten main feature groups exactly partition all 389 rich columns;
  diagnostic history subgroups deliberately overlap.
