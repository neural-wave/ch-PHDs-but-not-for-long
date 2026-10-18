---
title: 'Lag HPO completed: prior hierarchical winner retained and conformal audited'
status: current
kind: result
recorded_at: '2026-10-16T21:28:55.538490+00:00'
sources:
- results/hierarchical_hpo_lags_vast/metrics.json
- HIERARCHICAL_PROTOCOL.md
experiment_id: null
supersedes: []
---

2026-10-16 live audit: Nightwatch10e4974b8b1d9737a60aac8e completed exit0 at23:07:59 Zurich (277.80s from23:03:21). Remote /workspace/HACKATON/results/hierarchical_hpo_lags_vast/metrics.json and worker log confirm24/24 trials across0/1/3/7/14/28 daily lags completed. None beat frozen parent on August1–15 selection, so parent_selected retained: hierarchical32/two-block original389-feature history recipe, selectionMAE4.5182432717, full-August descriptive4.3836934611, September5.2827771386. Prior monthly joint blend5.091209 remains better; no manifest change. Do not choose another candidate by lower September score: September is development test and model selection used only declared selection dates. Lag comparison is randomized with other hyperparameters varied; it does not isolate causal effect of lag count.

All neural serialized replays passed, final offline point/interval replay maxerror9.834766331e-6; CSV11520 finite unique rows passed, hourly mean maxerror5.684341886e-14. Marginal conformal nominal50/90/95 actual September42.3264/86.4931/92.9340 percent. Hour-joint actual39.8611/86.4236/91.9444 percent. Day-joint nominal50/90 actual46.6667/76.6667 percent. Dependence/drift prevents guarantee; bands quantify uncertainty but under-cover. Metrics, testNPZ and parent_selected.pt retrieval initiated into local results/hierarchical_hpo_lags_vast. Next work optional scientific followup; no active Vast campaign remains from this request. Ando deployment refit was not rechecked.
