---
title: Architecture explanation and hourly-mean versus MAE clarification
status: current
kind: analysis
recorded_at: '2026-10-16T20:53:52.855674+00:00'
sources:
- submission/evaluation_audit.json
- submission/deployment_audit.json
- submission/frozen_recipe.json
- submission/graph_models.py
- submission/revin_models.py
- hierarchical_models.py
- HIERARCHICAL_PROTOCOL.md
experiment_id: null
supersedes: []
---

2026-10-16 user asked whether all models pass the mean constraint and requested deeper architecture explanation. Rechecked submission/{evaluation_audit.json,deployment_audit.json,frozen_recipe.json,predict.py,reconstruction.py,graph_models.py,revin_models.py} and current hierarchical_models.py/HIERARCHICAL_PROTOCOL.md. Completed HTML models enforce hourly mean by output zero-mean residuals plus final projection; evaluation exported max mean error2.51248e-11 and deployment tested max2.50111e-11 versus required1e-4 EUR/MWh. MAE is separate accuracy criterion, not a pass threshold. Audits concern the saved evaluation and final deployment artifacts; do not claim every unused historical candidate received independent full-month replay.

Delivered mixture has23 leaf entries:3 ridge,1 forest,4 hist boost configurations (four quarter regressors each),1 CatBoost configuration (four quarter models),8 MLP,4 temporal/node graph,1 hybrid RevIN and1 flat deterministic component. Approx family weights by Zone0/1/2/3: MLP38.0/39.5/20.3/46.9%; graph15.9/16.3/25.6/14.0%; RevIN3.2/12.2/11.1/0.0%. Graph processes batch×4×24×389, linear width64/128, one/two blocks of two width-preserving kernel3 temporal convs, four-head cross-zone attention per hour, width→2width→width FFN, residual LayerNorm, final four residuals with zero mean. All-day curves are already in rich features. RevIN adds7 channels to389, normalizes supplied24-hour curves perday/zone, preserves original rich features and mean/scale side channels, reverses residual scale. Hierarchy code independently maps contrasts a,u,v to a+u,a-u,-a+v,-a-v; its point mean is guaranteed algebraically, interval endpoints need not conserve mean. Latest separate hierarchy/HPO runtime was not checked for this Q&A, and its pending startup context was preserved.
