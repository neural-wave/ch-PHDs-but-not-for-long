---
title: RevIN assessment and completed joint-node comparison
status: superseded
kind: analysis
recorded_at: '2026-10-16T20:25:29.564935+00:00'
sources:
- train_graph.py
- graph_models.py
- results/monthly_graph_gpu1/metrics.json
- results/monthly_best.json
experiment_id: null
supersedes: []
status_updated_at: '2026-10-16T20:29:30.242432+00:00'
---

Verified 2026-10-16: train_graph.py uses fixed parent training-set mu/sd and clipped feature standardization. graph_models.py uses LayerNorm inside blocks; neither is RevIN. Current models predict zero-mean residuals over mean-compliant interpolation. User asked whether RevIN is used and useful. Answer: no; promising controlled ablation for day/zone level and volatility shifts, untested here. Recommended statistics computed from only each zone's supplied 24 hourly prices; normalize input and reverse output scale, retaining absolute level/scale and cross-zone spread information, then enforce hourly means. Select on August, no target-derived statistics. Original implementation/reference: https://github.com/ts-kim/RevIN . No RevIN experiment launched.

Monthly joint-node comparison completed: results/monthly_graph_gpu1/metrics.json August MAE 4.24743872379048; September development-test MAE 5.09226969906224. Winner model.pkl retrieved locally; 302 training days. Nightwatch a08c4ed9d064856b2beaf5bd completed exit 0. Whole-month CPU audit currently incomplete: arbitrary pointwise replay assertion 0.0002 failed at 0.0018328161 EUR/MWh, despite recomputed MAE 5.092269863919437 and hourly constraints passing. Precision diagnostic on Ando GPU1 graph_8: GPU/CPU residual max difference 0.00804424 with cudnn TF32 on versus 0.0000190735 with TF32 off; matmul TF32 false. Remote evidence /home/mnogales/Projects/HACKATON/results/monthly_graph_gpu1/{numerical_diagnostics.json,precision_diagnostics.json,audit_cpu_predictions.npz}; scripts graph_campaign/{audit_monthly_model.py,diagnose_graph_precision.py}. Audit must record numerical tolerance explicitly rather than claim exact replay. Broad goal remains active.
