# Frozen champion coherent-input sensitivity

User requested explainability of the champion on 17 October 2026. This study extends existing ensemble/conditional diagnostics to underlying inputs. No refitting or predictive-model promotion.

Use exact `minimize_expanding_20261017/fixed_blend_ratio` checkpoints, selected seed/family weights and fixed preprocessing for each of eight folds. Check source hashes, regenerate original features, replay all eleven quantiles within 1e-6 EUR/MWh, verify strictly prior history by future perturbation and history truncation, and retain finite ordered quantiles and conserved q50 means for every intervention. Store checkpoint hashes. Missing artifacts or failed replay stop execution.

Prespecified sample: days 3, 10, 17 and 24 of each February–September test month, all zones, hours and quarters. No error-based sampling. This is a 32-date diagnostic, not exhaustive attribution.

History interventions replace each strictly prior day's quarters by `historical_interpolation + s*(observed_quarters-historical_interpolation)`, for s=0.5 and 0. Preserve every historical hourly mean and today's supplied hourly inputs. Rebuild all dependent features and analog retrieval. Report forecast/correction change and original-target MAE change, with masking/distribution-shift caveats. Reference uses each historical day's available hourly curve, never test targets.

Current-curve interventions scale today's hourly deviations from each zone's daily mean by 0.9 and 1.1, with observed history unchanged. Rebuild dependent features/analog retrieval and interpolation anchor. Report forecast, anchor and learned-correction changes separately. Do not score against original quarters as counterfactual truth. Coherent synthetic probes are not established physical counterfactuals.

Aggregate changes with equal month weights; retain monthly results, fixed example ZONE_0 hour 18 on first sampled date, hashes and audit. No feature-importance ranking across unequal intervention sizes, causal effects, feature-necessity conclusions or generalization guarantees. Existing interval calibration is not validated for altered inputs. All months are dependent developmental evidence. No classifier work, submission edits or training.

Run through `python run.py campaigns/champion_sensitivity_20261017/analyze.py` on authorized research GPU, supervised by Nightwatch with success/failure reporting. Outputs: `results/champion_sensitivity_20261017/`.
