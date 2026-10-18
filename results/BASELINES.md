# Reconstruction baselines (2026-10-16)

Evaluated all 139,392 supplied observations: 363 days, 4 zones, 96 quarters per day/zone. These are descriptive training-set reconstruction errors, not secret-test results. No fitting or parameter selection was performed.

The challenge PDF sections 9–10 define MAE per day/zone, then average days and equally weight zones. Each reconstructed hour must retain its input mean within 0.0001 EUR/MWh. The README defines the hourly input as the arithmetic mean of four observed quarters.

| Method | MAE (EUR/MWh) | RMSE (EUR/MWh) | Hourly mean compliant |
|---|---:|---:|---|
| Repeat hourly mean | 7.577610 | 13.109974 | Yes |
| Linear interpolation | 5.738219 | 9.782215 | No |
| Linear interpolation + mean correction | 5.474193 | 9.448924 | Yes |

Interpolation treats hourly observations as values at hour centres (00:30, 01:30, …), and evaluates at quarter centres (00:07:30, 00:22:30, …). Each date and zone is independent. Values outside the first/last hour centre are held constant; no next-day observations are used. Connecting the observed quarter-hour targets would yield a trivial zero error and is not the reconstruction baseline evaluated here.

Mean correction adds H(h) minus the mean of the four interpolated values to each predicted quarter in hour h. Plain interpolation's maximum hourly-mean deviation is 58.5528125 EUR/MWh; corrected interpolation's maximum is approximately 1.14e-13. Corrected MAE is 27.76% lower than the flat baseline.

Run `python3 evaluate_baselines.py` from this directory or any working directory. It validates unique complete day/zone grids, finite values and expected counts, then writes results/baseline_metrics.json, results/baseline_daily_metrics.csv and results/baseline_predictions.csv. Full-precision predictions are retained; no two-decimal output rounding is applied.
