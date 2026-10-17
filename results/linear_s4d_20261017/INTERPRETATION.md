# One-layer S4D interpretation

All 128 fits and both independent accuracy/interpretability audits completed.
Eight expanding February–September developmental folds follow VALIDATION_PROTOCOL.md.

Learning frequencies reduced the equal-month MAE ratio from 0.817796 to 0.810498
(about 0.89% relative improvement over frozen frequencies). Pooled MAE was
4.698031 EUR/MWh; calibrated 90% coverage was 90.79% at mean width24.309 EUR/MWh.
The previous matched nonlinear temporal CNN scored0.778942, so the simpler model
trades some accuracy for a much smaller and exactly decomposable point path.

## What the frequencies show

The median absolute change from declared initial period anchors was3.22%,
unweighted across learned model modes/channels/folds. This is mostly adjustment
of an initialized filter bank, not evidence of discovering physical cycles
without prior anchors. Across the selected learned models, median periods near
initial12/24-hour anchors were11.91/23.47hours. Those anchors were explicitly
supplied at initialization and must retain that provenance.

Long-period modes varied widely: the168-hour initialized band had median178.7h,
range98.1h to282807h. Near-zero discrete angles correspond to nearly nonoscillatory
modes over this window, not observed multi-year seasonality. This does not imply
unstable dynamics: real pole parts remain negative. Seven days provide limited
identification of weekly and longer oscillations.

Mode amplitude markers are a point-readout coefficient proxy, not causal feature
importance or a measured physical spectral power. Hourly conservation can cancel
slow/common components; inspect the exported effective kernels and exact
contribution curves, not modal labels alone. No statistical significance claim.

## Exact explanation evidence

The point path is affine given availability masks. All16 family/fold winners
passed source-channel contribution summation and independent NumPy convolution
replay. Each first held-out day has baseline, coherent point curve, ten additive
input-channel contributions, normalizers, lag kernels/feedthrough and effective
normalized-input kernels at hours0/12/23. These explain the model exactly; they
do not establish a causal account of market prices. Uncertainty uses a separate
ordered softplus head and empirical reserved-block conformal widening.

Artifacts: frequencies.csv,frequency_summary.json,frequencies.png,
folds/*/interpretations/*/exact_explanation.npz,interpretability_audit.json.
