# Code index

Run `python run.py <script-name> [args]` from the repository root. This index classifies filenames for navigation; historical launchers require inspection before reuse.

## Submission preparation

- [check_submission.py](../code/check_submission.py): file/checksum and declared handoff-status audit; no model execution.
- [Report builder](../campaigns/submission_materials_20261017/build_report.py): three-page draft from editable Markdown and saved monthly metrics; optional authoring dependencies in `requirements-report.txt`.

## Models and shared utilities

- [add_conformal_50.py](../code/add_conformal_50.py)
- [catboost_models.py](../code/catboost_models.py)
- [expanding_simple.py](../code/expanding_simple.py)
- [frequency_experiment.py](../code/frequency_experiment.py)
- [graph_models.py](../code/graph_models.py)
- [hierarchical_features.py](../code/hierarchical_features.py)
- [hierarchical_models.py](../code/hierarchical_models.py)
- [interpolation_history.py](../code/interpolation_history.py)
- [interpretable_models.py](../code/interpretable_models.py)
- [joint_zone_cnn.py](../code/joint_zone_cnn.py)
- [linear_quantile_model.py](../code/linear_quantile_model.py)
- [parallel_zone_cnn.py](../code/parallel_zone_cnn.py)
- [quantile_cnn.py](../code/quantile_cnn.py)
- [reconstruction.py](../code/reconstruction.py)
- [revin_models.py](../code/revin_models.py)
- [segmented_frequency.py](../code/segmented_frequency.py)
- [simple_conformal.py](../code/simple_conformal.py)
- [simple_search.py](../code/simple_search.py)
- [ssm_experiment.py](../code/ssm_experiment.py)
- [ssm_models.py](../code/ssm_models.py)

## Analysis and reports

- [analyze_cnn_confidence.py](../code/analyze_cnn_confidence.py)
- [analyze_feature_importance.py](../code/analyze_feature_importance.py)
- [analyze_quantile_cnn.py](../code/analyze_quantile_cnn.py)
- [analyze_revin.py](../code/analyze_revin.py)
- [diagnose_graph_precision.py](../code/diagnose_graph_precision.py)
- [inspect_interpolation_difficulty.py](../code/inspect_interpolation_difficulty.py)
- [inspect_runs.py](../code/inspect_runs.py)
- [inspect_validation_risk.py](../code/inspect_validation_risk.py)
- [plot_error_hotspots.py](../code/plot_error_hotspots.py)
- [plot_expanding.py](../code/plot_expanding.py)
- [plot_linear_quantiles.py](../code/plot_linear_quantiles.py)
- [plot_monthly_baselines.py](../code/plot_monthly_baselines.py)
- [plot_monthly_baselines_by_zone.py](../code/plot_monthly_baselines_by_zone.py)
- [residual_diagnostics.py](../code/residual_diagnostics.py)
- [summarize_riccardo_comparison.py](../code/summarize_riccardo_comparison.py)
- [write_experiments_html.py](../code/write_experiments_html.py)

## Audits and verification

- [audit_deployment.py](../code/audit_deployment.py)
- [audit_expanding_simple.py](../code/audit_expanding_simple.py)
- [audit_frequency_experiment.py](../code/audit_frequency_experiment.py)
- [audit_joint_zone_cnn.py](../code/audit_joint_zone_cnn.py)
- [audit_monthly_model.py](../code/audit_monthly_model.py)
- [audit_quantile_cnn.py](../code/audit_quantile_cnn.py)
- [audit_quantile_results.py](../code/audit_quantile_results.py)
- [audit_riccardo_comparison.py](../code/audit_riccardo_comparison.py)
- [audit_segmented_frequency.py](../code/audit_segmented_frequency.py)
- [audit_simple_search.py](../code/audit_simple_search.py)
- [audit_simplicity.py](../code/audit_simplicity.py)
- [audit_ssm_experiment.py](../code/audit_ssm_experiment.py)
- [audit_tuned_parallel_cnn.py](../code/audit_tuned_parallel_cnn.py)
- [verify_graph.py](../code/verify_graph.py)
- [verify_hierarchical.py](../code/verify_hierarchical.py)
- [verify_interpolation_history.py](../code/verify_interpolation_history.py)
- [verify_linear_quantiles.py](../code/verify_linear_quantiles.py)
- [verify_projection.py](../code/verify_projection.py)
- [verify_revin.py](../code/verify_revin.py)
- [verify_rich_features.py](../code/verify_rich_features.py)

## Selection and packaging

- [build_submission.py](../code/build_submission.py)
- [export_best_research_recipe.py](../code/export_best_research_recipe.py)
- [export_frozen_recipe.py](../code/export_frozen_recipe.py)
- [export_rolling_plan.py](../code/export_rolling_plan.py)
- [finalize_deployment.py](../code/finalize_deployment.py)
- [rank_models.py](../code/rank_models.py)
- [refit_deployment.py](../code/refit_deployment.py)
- [select_best.py](../code/select_best.py)
- [select_monthly.py](../code/select_monthly.py)

## Evaluation

- [compare_riccardo.py](../code/compare_riccardo.py)
- [evaluate_baselines.py](../code/evaluate_baselines.py)
- [evaluate_normalized_validation.py](../code/evaluate_normalized_validation.py)
- [evaluate_september_baselines.py](../code/evaluate_september_baselines.py)

## Inference

- [predict.py](../code/predict.py)
- [predict_best.py](../code/predict_best.py)
- [predict_hierarchical.py](../code/predict_hierarchical.py)

## Historical launchers

- [run_catboost_after.py](../code/run_catboost_after.py)
- [run_deployment_refit.py](../code/run_deployment_refit.py)
- [run_expanding.py](../code/run_expanding.py)
- [run_expanding_simple.py](../code/run_expanding_simple.py)
- [run_frequency_experiment.py](../code/run_frequency_experiment.py)
- [run_graph_after.py](../code/run_graph_after.py)
- [run_hierarchical_hpo.py](../code/run_hierarchical_hpo.py)
- [run_hierarchical_vast.py](../code/run_hierarchical_vast.py)
- [run_joint_zone_cnn.py](../code/run_joint_zone_cnn.py)
- [run_linear_quantiles.py](../code/run_linear_quantiles.py)
- [run_monthly.py](../code/run_monthly.py)
- [run_parallel_zone_cnn.py](../code/run_parallel_zone_cnn.py)
- [run_projection_after.py](../code/run_projection_after.py)
- [run_quantile_cnn.py](../code/run_quantile_cnn.py)
- [run_revin.py](../code/run_revin.py)
- [run_riccardo_comparison.py](../code/run_riccardo_comparison.py)
- [run_rich_after.py](../code/run_rich_after.py)
- [run_rolling.py](../code/run_rolling.py)
- [run_segmented_frequency.py](../code/run_segmented_frequency.py)
- [run_simple_joint_zone_cnn.py](../code/run_simple_joint_zone_cnn.py)
- [run_simple_search.py](../code/run_simple_search.py)
- [run_ssm_experiment.py](../code/run_ssm_experiment.py)
- [run_tuned_parallel_cnn.py](../code/run_tuned_parallel_cnn.py)
- [run_zone_search.py](../code/run_zone_search.py)

## Training and tuning

- [train_catboost.py](../code/train_catboost.py)
- [train_expanding.py](../code/train_expanding.py)
- [train_frozen.py](../code/train_frozen.py)
- [train_graph.py](../code/train_graph.py)
- [train_hierarchical.py](../code/train_hierarchical.py)
- [train_interpretable.py](../code/train_interpretable.py)
- [train_joint_blend.py](../code/train_joint_blend.py)
- [train_linear_quantiles.py](../code/train_linear_quantiles.py)
- [train_revin.py](../code/train_revin.py)
- [train_rolling.py](../code/train_rolling.py)
- [train_search.py](../code/train_search.py)
- [tune_parallel_cnn.py](../code/tune_parallel_cnn.py)

## Quantiles and RevIN

- [Features, model explanations and band plots](MODELS_AND_UNCERTAINTY.md)
- [Saved champion quantile examples](../code/plot_quantile_examples.py)
- [Daily-normalized display of saved predictions](../code/plot_revin_examples.py)
- [Matched compact CNN RevIN trial](../campaigns/revin_quantiles_20261017/PROTOCOL.md)
- [Champion ensemble RevIN refits](../campaigns/champion_revin_20261017/PROTOCOL.md)
- [Standalone linear-compact RevIN refits](../campaigns/linear_compact_revin_20261017/PROTOCOL.md)
