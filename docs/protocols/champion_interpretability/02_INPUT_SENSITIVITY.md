# 2. Grouped input sensitivity

Question: does the champion respond mainly to current ramps, cross-zone context or earlier quarter shapes?

Define groups from the actual feature code: current hourly curves, cross-zone context, calendar, lagged quarter shapes, lagged interpolation residuals and similar-day residuals. Perturb underlying inputs in coherent date/zone blocks using fitting-only reference distributions; regenerate every dependent engineered feature and analog retrieval. Avoid independent column permutation of redundant prices. Calendar interventions must represent valid dates and consistent history.

Use two explicitly separated diagnostics: historical-shape masking with current hourly means unchanged; plausible current-curve perturbations with a recomputed interpolation anchor. For the latter split delta forecast into delta baseline plus delta learned correction. Original observed quarters are not counterfactual targets. Measure forecast change in EUR/MWh and contrast coordinates; report original-target MAE degradation only for fixed-hourly history interventions, with distribution-shift caveats.

Optional grouped integrated gradients explain differentiable model stages, with training-only references and completeness checks; discrete analog retrieval and clipping need separate treatment. These are model sensitivities, not physical causal effects. If testing feature necessity by retraining, create a separate matched-budget eight-fold campaign with fresh selection/calibration.

Acceptance: coherent perturbation manifest, no future quarters, dependent features rebuilt, hourly/quantile checks pass, results separated by intervention type and month.

See [shared protocol](PROTOCOL.md). Status: proposed, not executed.
