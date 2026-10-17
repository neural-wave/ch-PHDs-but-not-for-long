---
title: Launched exact champion ensemble refits with RevIN and remade full-day plot
  pipeline
status: current
kind: experiment
recorded_at: '2026-10-17T18:08:17.957091+00:00'
sources:
- campaigns/champion_revin_20261017/PROTOCOL.md
- campaigns/champion_revin_20261017/frozen_components.json
- results/champion_revin_20261017/launch.json
experiment_id: null
supersedes: []
---

User explicitly requested train best model with RevIN and remake last full-day quantile plot. Owning registry best_model.json identifies minimize_expanding_20261017/fixed_blend_ratio ratio0.7079682867567635. Frozen exact fold recipes from results/model_ranking_20261017/champion_per_fold.json copied campaigns/champion_revin_20261017/frozen_components.json (51leaves total). New campaign fresh-fits those exact architectures/lr/seeds with learned-affine input-day RevIN, preserving convex blend weights and original ratio-weighted physical MAE+pinball loss; prior-selection stopping epochs can change. This is frozen-recipe transfer, not new full candidate search; developmental original recipe. Uses input-day mu/s on current/past source prices; preserves original causal analog retrieval weights/outputs then rescales residuals; retains existing mean/std side features, categorical masks, fitting-only knots. Shared positive scalar gamma/beta per expert; inverse gamma then per-zone scale and original interpolation anchor. User requested plotted March14ZONE2/April26ZONE3; pipeline makes normalized shared axes plus EUR/MWh versions, original champion left/new RevIN right, all11quantile lines/calibrated90bands. Remote all-six-family negative/flat, nonidentity affine, gradient/day-isolation, order/conservation and loss-unit edge checks passed; compatibility fix converts history dates to Python strings without modifying original source. Nightwatch health and idle RTX5090 verified. Registered/detached job cbf0b93e1888ddd4d52e0d11 worker15184 on root@93.91.156.85:55340 /workspace/HACKATON, log/root/.local/share/nightwatch-hackaton/cbf0b93e1888ddd4d52e0d11.log. Remote results/workspace/HACKATON/results/champion_revin_20261017. ETA20minutes, success/crash Telegram existing owner authorized. Success gated on first/last5-checkpoint smoke, all51replay/CQR/chronology/weights/sourceidentity audit, originalchampion score recovery, full-day plots/report/archive. No deployment changes. Source protocol, new scripts, launch receipt local; original source/checkpoints preserved. Previous startup verbatim memory/records/20261017_startup_before_champion_revin.md. Job pending; no fit/plot completion claimed.
