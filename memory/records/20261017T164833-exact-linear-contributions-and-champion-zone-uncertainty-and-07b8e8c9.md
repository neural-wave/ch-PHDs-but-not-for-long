---
title: Exact linear contributions and champion zone, uncertainty and cancellation
  explanations
status: current
kind: experiment
recorded_at: '2026-10-17T16:48:33.747065+00:00'
sources:
- results/interpretability_deep_20261017/REPORT.md
- results/interpretability_deep_20261017/audit.json
- results/interpretability_deep_20261017/provenance.json
- campaigns/interpretability_deep_20261017/PROTOCOL.md
- docs/SUBMISSION_REFIT_PLAN.md
experiment_id: null
supersedes: []
---

Completed 2026-10-17 on root@93.91.156.85:55340 /workspace/HACKATON. Nightwatch 9bdcaa1b892f8998f25b2959 delivered finish exit0; log /root/.local/share/nightwatch-hackaton/9bdcaa1b892f8998f25b2959.log. No training or model/submission changes. Exact replay of standalone September linear JSON matches saved MAE5.2261054787 with zero difference, mean error5.68e-14. Actual grouped contribution magnitudes: analogous historical interpolation residuals3.410, slopes2.055 EUR/MWh; these are descriptive contributions, not causal/ablation utility. Exact success/failure feature and quarter decompositions saved and figures visually inspected. Current champion eight-fold forecast blend replay discrepancy<=1.14e-13; primary ratio .707968 reproduced. Weighted expert error ratio .728740, blend .707968: cancellation .020771 ratio units, ~.125 EUR/MWh pooled; opposite-sign quarters21.28%, blend beats every active expert whole-hour MAE on8.03% of hours (equal-month means). Current curve gains falling33.66/rising32.29/peak23.34/trough10.59/flat8.69%. Same-curve-mix zone-standardized ratios .59907/.65567/.82040/.78295 retain zone gap. ZONE_2 best single analogCNN ratio .823397 versus blend .822717, five of eight month wins. Uncertainty claim recomputed on identical monthly subsets: ratios .88550/.75100/.64647 ->11.45/24.90/35.35% reductions, not ratio of mean MAEs. Member-removal variants are uncalibrated posthoc diagnostics, not new selected policies. User states submission must train on all available months: recorded docs/SUBMISSION_REFIT_PLAN.md (363 eligible days; freeze recipe first, causal per-example histories; final quantile calibration transfer requires separate chronological verification). No all-month refit launched. No blind or independent-fold significance claim.
