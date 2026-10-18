---
title: Clarified earlier ensemble and exact conformal calibration interpretation
status: current
kind: clarification
recorded_at: '2026-10-16T21:31:17.964931+00:00'
sources:
- results/monthly_joint_blend_gpu1/metrics.json
- results/monthly_best.json
- train_joint_blend.py
- train_hierarchical.py
- results/hierarchical_hpo_lags_vast/metrics.json
experiment_id: null
supersedes: []
---

2026-10-16 user asked what earlier5.0912 model is and how conformal/coverage works. Verified owning artifacts: monthly_joint_blend_gpu1 is a zone-specific convex ensemble built by reweighting selected base ridge/tree/MLP, CatBoost, joint temporal-attention and RevIN components with frozen parent fallback, optimized on all31 August dates. SeptemberMAE5.09120881999 vs hierarchy5.28277713863. Same302 fitting days/September targets, different selection budget: hierarchy uses15 August dates and16 exclusively for calibration. Thus this compares actual achieved scores, not a controlled architecture-only test. No conformal calibration has been applied here to the old ensemble using that reused August data.

Exact marginal conformal recipe: freeze model after Aug1–15 selection; compute absolute quarter errors Aug16–31, pool16*24=384 hours separately per zone and quarter position (16 groups); coverage c uses sorted residual rank ceil((384+1)*c), interval yhat±radius. Fixed radii by zone/quarter position, not input-adaptive, no online September recalibration.50/90/95 nominal coverage has actual42.3264/86.4931/92.9340 percent across11520 September quarters. These figures are marginal, not simultaneous day/hour coverage and not confidence about model parameters. Hour-joint uses max-four-quarter residual per zone/hour; day-joint uses max384 quarter residual per complete day with16 calibration blocks. Point hourly means remain exact; interval endpoints are envelopes and need not average to hourmean. Undercoverage consistent with larger/changed future residuals but drift vs dependency cause is not isolated. Reference primary literature https://proceedings.mlr.press/v313/barber26a.html. No new training or external deployment requested.
