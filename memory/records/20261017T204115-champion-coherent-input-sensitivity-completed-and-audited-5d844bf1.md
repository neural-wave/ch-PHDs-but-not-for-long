---
title: Champion coherent-input sensitivity completed and audited
status: current
kind: experiment
recorded_at: '2026-10-17T20:41:15.548933+00:00'
sources:
- campaigns/champion_sensitivity_20261017/PROTOCOL.md
- results/champion_sensitivity_20261017/audit.json
- results/champion_sensitivity_20261017/summary.json
- results/champion_sensitivity_20261017/REPORT.md
experiment_id: null
supersedes: []
---

User requested explainability on actual champion, interpolation clarification and exact ensemble claim. Implemented campaigns/champion_sensitivity_20261017/analyze.py and PROTOCOL.md; ran on root@93.91.156.85:55340 in /workspace/HACKATON. Nightwatch 005ba052519b64fb44c2a9f1 completed exit 0 with delivered success; log /root/.local/share/nightwatch-hackaton/005ba052519b64fb44c2a9f1.log. Recovered/visually inspected results/champion_sensitivity_20261017/. Exact eight frozen champion blends replay all eleven quantiles within 1.14e-13 EUR/MWh; source hashes, causal probes, output checks passed. Prespecified four whole dates/month (3,10,17,24), 32 dates. Historical residuals relative to historical interpolation scaled by 0.5/0, preserving hourly means and rebuilding dependent features/analogs. Zero residuals changes q50 2.35545 EUR/MWh and raises equal-month sampled MAE by 0.64836 (3.88856 to 4.53691), worse in all eight sampled months; half residuals raises MAE 0.13350. Current curves +/-10% about zone daily means changes learned correction 0.31632/0.30863 after anchor subtraction; original truth not scored for changed hourly inputs. This is fixed-checkpoint sensitivity with distribution-shift caveats, not retrained ablation, causal attribution, exhaustive test coverage or comparable feature ranking. Initial two Nightwatch attempts 470a8bfd38d94cbee6479f5c and ea101bbfcfc389ee8ea3c5f4 failed replay because reduced GPU batch altered rounding by 1.15e-5; fixed by preserving original inference batch shapes, retaining 1e-6 tolerance. No model/submission edits. Existing submission refit Nightwatch 2b509aeec0a930d6a9979619 finish exit 0/delivered observed, but submission audits/recovery not independently reviewed here.
