# Frozen champion input sensitivity

Completed on authorized research GPU. Four prespecified dates (3, 10, 17, 24) per test month, all zones/hours. No fitting, recipe selection or submission changes.

Original checkpoint blends replayed all eleven saved quantiles within 1e-6 EUR/MWh. Source hashes and causal feature probes passed. Every intervention retained ordered finite quantiles and conserved q50 hourly means.

| Intervention | Forecast change | Learned-correction change | Original-target MAE change |
|---|---:|---:|---:|
| history_residual_half | 1.1727 | 1.1727 | +0.1335 |
| history_residual_zero | 2.3555 | 2.3555 | +0.6484 |
| current_curve_90pct | 3.4244 | 0.3163 | Not a counterfactual target |
| current_curve_110pct | 3.4211 | 0.3086 | Not a counterfactual target |

Values are equal-month means in EUR/MWh. History interventions shrink observed historical deviations from mean-corrected interpolation while preserving all historical hourly means. Current-curve interventions scale deviations from each zone’s daily mean with history unchanged; the anchor and all dependent features/analog retrieval are rebuilt.

Historical masking can be out of distribution. Curves are synthetic coherent sensitivity probes, not validated physical counterfactuals. Intervention sizes differ: do not rank input importance from these magnitudes. Saved interval calibration is not revalidated for modified inputs. Test months are developmental, dependent and already inspected. Fixed dates are an economical diagnostic, not exhaustive attribution.

Evidence: protocol.json, monthly.csv, summary.json, examples.json, audit.json. Exact frozen checkpoint paths/hashes in audit.json. Reproduce: python run.py campaigns/champion_sensitivity_20261017/analyze.py on research host.
