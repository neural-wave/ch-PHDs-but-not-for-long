---
title: Literature-informed evaluation recommendation
status: superseded
kind: recommendation
recorded_at: '2026-10-17T08:38:22.658226+00:00'
sources:
- EXPANDING_PROTOCOL.md
- SIMPLE_SEARCH_PROTOCOL.md
- DEPLOYMENT_REFIT.md
experiment_id: null
supersedes: []
status_updated_at: '2026-10-17T13:47:55.400344+00:00'
---

User asked which evaluation is best and explicitly requested web research on 2026-10-17. Recommendation, not an executed or accepted protocol change: monthly walk-forward outer tests with chronological inner selection, followed by genuinely unseen official challenge evaluation if available. Existing three-training-month start is a pragmatic coverage compromise, not a literature-established optimum; one/two-month folds can be cold-start diagnostics. Full nested version uses earlier rolling validation splits within each outer pre-test history to select the entire pipeline (features/model/window/ensemble weights/epochs), then refits per a predeclared rule and freezes weights through the outer test month. Single preceding validation month is a simpler defensible version already implemented. Evaluate expanding versus fixed recent training windows on identical outer test dates, select windows only using inner validation. Match offline reconstruction inputs: known hourly curves permitted; quarter targets only as historically available. Keep all zones and quarters of each day together; train preprocessing separately at every fit. Report pooled target-weighted quarter MAE, monthly/per-zone MAE and same-date baseline comparison; keep September separately visible. If uncertainty is added, use paired time blocks preserving all zones, not independent quarter resampling. Previously inspected months remain developmental; nested resampling cannot erase retrospective human model-selection exposure. No experiments launched.

Primary sources consulted: Cerqueira/Torgo/Mozetic, Evaluating time series forecasting models (Machine Learning 2020), https://arxiv.org/abs/1905.11744: empirical support for temporally ordered OOS evaluation under nonstationarity, no universal best method. Hyndman/Athanasopoulos FPP3 time-series CV, https://otexts.com/fpp3/tscv.html: rolling-origin evaluation and adequate initial training. Official scikit-learn TimeSeriesSplit https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html and nested-CV explanation https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html: separating tuning and evaluation avoids optimistic selection scores. Recommendation is tailored inference; these sources do not validate a specific three-month minimum or electricity-specific training window.
