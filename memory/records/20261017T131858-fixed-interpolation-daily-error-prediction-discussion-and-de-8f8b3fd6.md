---
title: Fixed interpolation daily error prediction discussion and descriptive signal
  20261017
status: current
kind: design
recorded_at: '2026-10-17T13:18:58.394995+00:00'
sources:
- quantile_cnn.py
- reconstruction.py
- INTERPOLATION_HISTORY_PROTOCOL.md
- inspect_interpolation_difficulty.py
- results/quantile_2d_cnn_20261017/interpolation_difficulty_diagnostic.json
experiment_id: null
supersedes: []
---

User asked whether we can predict linear interpolation error given the day, and requested an opinion. Ambiguity is important: predicting the full signed quarter-hour residual curve is already implemented by our CNN (quantile_cnn.py targets=(y-b)/20, reconstruction=b+centered predicted residual). Predicting one scalar daily absolute interpolation error is a separate proposed target. No new predictor or experiment launched in this turn, and the earlier decision to set aside CNN-confidence/catastrophe detection remains intact.

Assuming the proposed scalar target uses the known full four-zone hourly curves, define fixed mean-preserving interpolation B(H) and E_day=mean over zones/quarters |Y-B(H)|. Inputs can use query hourly prices/ramps/curvature, calendar, and strictly earlier interpolation-error histories. Date alone is a weaker input specification. Do not use current true residuals/MAE as features. Because B is fixed and untrained, valid labels can be computed directly for every earlier training day; no refitting a CNN for out-of-sample error labels is necessary. Chronological training/selection/calibration separation still applies. A scalar magnitude estimate does not provide the signed correction needed to improve individual quarter prices.

Recomputed descriptive evidence from official data.npz and fixed baseline b. Training302days: interpolation dailyMAE mean5.235243, range1.975931–11.895199,7days>10. Correlation with mean absolute current-day hourly ramp Pearson.826069; mean absolute second difference.836624. August1–15 baselineMAE6.732429, one day>10(Aug13,10.095251); August16–31 baselineMAE5.774529, zero>10. September baselineMAE7.090321, range4.681540–9.871533, zero>10; ramp correlation.583053, curvature.687148. The signed baseline residual sums to zero within each hour to numerical precision(max5.684e-13). These are associations, not fitted-predictor results, and September is already developmental.

Recommendation discussed: bounded simple continuous daily-MAE regression using current hourly shape and prior interpolation residual magnitudes; compare against a constant baseline and simple hourly-ramp features. Evaluate actual scalar prediction error, calibration and ranking on chronological held-out data; avoid the sparsely supported10EUR/MWh binary catastrophe target. This could flag difficult interpolation days, but it does not itself improve reconstructed prices or validate error detection for the learned CNN. Full residual prediction would be a continuation of the existing reconstruction objective rather than a new target. Preserve parallel FlowState/other-chat work and deployment.
