# Final deployment refit

The validation-selected evaluation pipeline is monthly_joint_blend_gpu1: train302 days through July, August MAE4.245329, September development-test MAE5.091209. Its source model and whole-month serialized replay remain preserved. September was previously viewed.

A separate deployment_all363_gpu1 copy refits every unique selected learned component from scratch on all363 official dates, including August and September. Recompute feature statistics on all363; keep every selected model configuration, selected epoch/tree count, and ensemble weight frozen. No validation or test score controls this fit. Existing input-only RevIN statistics remain per supplied day/zone. TF32 is off. Tree/ridge/boosting work runs remotely; neural and CatBoost fits use physical Ando GPU1. Deduplicate shared serialized leaf artifacts to avoid duplicate fitting.

This is final training for unavailable secret challenge dates. It does not have a held-out September score. Reported August/September accuracy continues to refer exclusively to the preserved302-day evaluation model.

Source: code/refit_deployment.py and code/run_deployment_refit.py; remote snapshot revin_campaign/. A shortened all-family smoke must pass before the full refit. Finalization verifies unchanged model names/settings/weights, all363 fit provenance, saved-model exported finite/unique384 predictions and hourly mean tolerance. It then writes results/deployment_best.json. The default prediction manifest remains the evaluated monthly pipeline; explicitly choose the deployment manifest when using the final fit.

Nightwatch816b1393f18c891ba8a8de11, supervisor2183339, Ando, initial ETA45minutes from20:34UTC October16. Log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/816b1393f18c891ba8a8de11.log. Verify terminal state and refit_report.json before claiming readiness.

## Independent finalization

The challenge requires training from scratch without existing fitted artifacts. `code/export_frozen_recipe.py` exports the effective convex leaf weights/settings to `frozen_recipe.json` (23 leaf entries including one deterministic flat component). `code/train_frozen.py` constructs fresh ridge/forest/boosting/neural components and trains directly from the official CSV; it requires no evaluation pickle or prediction cache. The deployment artifact itself remains the previously frozen nested fit until its independent audit passes.

`code/audit_deployment.py` checks all-363 normalization at every nested level, all forest root training counts, fixed boost/tree counts, finite learned states, unchanged source/settings/weights, and agreement with the frozen recipe. It tests shuffled official input rows, negative constant inputs with empty history, and a synthetic volatile curve. Synthetic eligibility tests reuse an existing training date; they do not access October prices or claim prediction accuracy. Equivalent flattening of nested convex weights must match within1e-7 EUR/MWh on these cases.

`code/build_submission.py` prepares an offline inference folder with the flattened artifact, complete from-scratch training code, frozen recipe, dependencies, methodology/evaluation notes and SHA256 manifest. Its packaged entry point is tested against audited predictions before success is reported. Preparation of this folder does not send it to the jury or freeze external submissions.

Queued finalization Nightwatch76cb026d6d5a2f3593617af3, supervisor2188716, initial20-minute ETA from20:41UTC. It waits for the verified refit PID2184735, requires a successful full report, runs all-family from-scratch recipe smoke, independent artifact audit, and package QA. Log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/76cb026d6d5a2f3593617af3.log. Both execution-host wrappers have heartbeats and actual success/crash alerts. Verify actual terminal results before retrieving or claiming final readiness.

## Verified final state

Refit Nightwatch816b1393f18c891ba8a8de11 and finalization76cb026d6d5a2f3593617af3 both completed exit0. The delivered folder is submission/. All16 delivered file hashes were verified locally after transfer. Nested all363 model SHA2442b10b4fcafb2c0c83cf2337bb05013dab3ba0df4b4bdb79be0dbef4e73a60; delivered flattened model SHA8ccb3b8346b9800d911546cdb6e2476e64fa89c26f7f250dd55f842e11d75805. Normalization, forest counts, settings/weights and recipe match passed. Three eligibility cases passed; maximum mean error<3e-11 EUR/MWh and equivalent flattening difference<3e-13. Packaged inference output matched the audited nested artifact. Full22-component from-scratch recipe smoke passed. The folder has no external submission side effect. See submission/README.md and FREEZE_MANIFEST.json.
