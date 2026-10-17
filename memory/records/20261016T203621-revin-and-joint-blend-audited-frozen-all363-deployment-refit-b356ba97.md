---
title: RevIN and joint blend audited; frozen all363 deployment refit running
status: superseded
kind: experiment
recorded_at: '2026-10-16T20:36:21.525636+00:00'
sources:
- REVIN_PROTOCOL.md
- DEPLOYMENT_REFIT.md
- results/monthly_revin_gpu1/revin_analysis.json
- results/monthly_joint_blend_gpu1/audit.json
- results/monthly_best.json
- refit_deployment.py
- results/experiments_summary.html
experiment_id: null
supersedes: []
status_updated_at: '2026-10-16T20:44:52.958582+00:00'
---

Verified 2026-10-16: RevIN Nightwatch845d62f24e76c25572da6009 completed exit0, eight candidates. Hybrid RevIN wins two of four matched pairs, average standalone August difference -0.006642. Selected revin_7 contribution improves parent ensemble August4.247439 to4.246177; September5.092270 to5.091086. Descriptive day-bootstrap September difference95% [-0.007020,+0.004205] spans zero; not valid post-selection significance. No broad RevIN superiority claimed. Full serialized September audit passed and local SHA20b0c460f32b8cadcb1d16ea4a4e7fd44917b444f689075c56e01d2e3edaa76d matches remote.

Joint leaf reweighting Nightwatchdfc5c8021bec7248371f2f87 completed exit0. 24 candidate components including frozen parent fallback, August-only convex and zone shrink0.05. August4.245328588334859, September5.091208819992601; selected despite slightly worse September than RevIN because August is sole selection criterion. Whole-month serialized audit passed: recomputed September5.0912092804265825, max exported mean error2.51248e-11, five nested bundle levels, forest counts28992; documented sub-cent TF32 replay tolerance retained. Local model SHA6615d87e9a07520572f58ffca75f164f8ae021140b700223c6a97df740810dcc verified. Manifest results/monthly_best.json updated locally/remotely; short HTML reflects five monthly comparisons and mixed RevIN evidence.

Final separate deployment refit implemented: refit_deployment.py/run_deployment_refit.py, protocol DEPLOYMENT_REFIT.md. All363 official dates, normalization recomputed, every unique learned leaf refit from scratch, selected settings/epochs/tree counts/weights frozen. No held-out score is claimed for this fit, and 302-day evaluation models remain intact. Deduplication across nested and shared leaf artifacts. All-family reduced smoke passed22 unique components and exported mean error2.5011e-11; full process2184735 running, CUDA_VISIBLE_DEVICES=1 verified directly. Nightwatch816b1393f18c891ba8a8de11 supervisor2183339, heartbeats/Telegram enabled, initial45min ETA23:19Zurich. Remote root /home/mnogales/Projects/HACKATON, sources revin_campaign/, outputs results/deployment_all363_gpu1, worker log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/816b1393f18c891ba8a8de11.log. On success writes refit_report.json and deployment_best.json. Default predict_best still monthly manifest; explicit deployment selection supported. Next: verify final refit terminal state and independent all363 artifact/provenance/inference audit, retrieve final artifact and freeze delivery package. Goal active.
