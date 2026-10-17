# 5. Explanation stability and optional surrogate

Question: which explanatory patterns persist across folds, zones and selected seeds?

Compare group sensitivities and expert contrast distributions across existing fits, conditioning on which experts are active. February/March membership differs, so report active-fold counts instead of a naive eight-month family ranking. For champion linear experts in their active folds, compute exact physical feature contributions including clipping, preprocessing, bias and projection; do not reuse the standalone September checkpoint as their explanation.

Assess signs and rankings within comparable feature groups; correlated features can exchange coefficients. Any date-block resampling remains descriptive and cannot remove developmental selection bias.

Optional: fit a small additive/linear surrogate to champion corrections on fitting data only, choose complexity on preceding-month days 1–15 and evaluate next-month fidelity. Report surrogate-to-champion MAE separately from truth MAE, contrast fidelity, monthly/zone fidelity and failure examples. A faithful surrogate describes the model, not the data-generating mechanism. If promoted as a predictor, it must satisfy the full quantile/calibration protocol.

Acceptance: original champion unchanged, source experts identified, membership differences explicit, surrogate fidelity evaluated outside its fitting/selection dates. Surrogate fitting belongs on a research server.

See [shared protocol](PROTOCOL.md). Status: proposed, not executed.
