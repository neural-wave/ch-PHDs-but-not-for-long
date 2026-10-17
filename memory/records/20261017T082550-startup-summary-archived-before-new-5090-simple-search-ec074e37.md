---
title: Startup summary archived before new 5090 simple search
status: historical
kind: continuity
recorded_at: '2026-10-17T08:25:50.419855+00:00'
sources:
- project_memory.md
experiment_id: null
supersedes: []
---

Preserved verbatim startup summary before shortening for the October 17 search:

# HACKATON project memory

Objective (2026-10-16): official-only quarter reconstruction, four zones, hourly means within 0.0001 EUR/MWh. No external/pretrained/October prices. Prior monthly winner: August 4.245329, September 5.091209; audits passed. Evidence results/monthly_best.json, results/monthly_joint_blend_gpu1/audit.json.

Ando GPU1 all363 deployment refit last seen running, not rechecked: Nightwatch816b1393f18c891ba8a8de11, supervisor2183339/process2184735; /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1. DEPLOYMENT_REFIT.md; no held-out deployment score.

Earlier Vast hierarchy/HPO/lag runs completed: September5.282777; marginal50/90/95 coverage42.33/86.49/92.93%. No systematic hierarchy gain in four matched pairs. User prefers interpretability: rich ridge5.294221, single MLP5.234445; ensemble gains3.83%/2.74%. Compact additive5.688563; coefficients/results in results/interpretable_vast. INTERPRETABLE_PROTOCOL.md.

Baseline zone charts preserved: results/monthly_baseline_mae_by_zone.png/.svg/.csv; zones anonymized, no country mapping. Detailed prior context: memory/records/20261016_before_expanding_startup.md.

New expanding evaluation and learning curves running on root@ssh1.vast.ai:24199, /workspace/HACKATON. Eight folds grow October–December training through July, validate next month, test following month (February–September). Eight additional folds hold July train-end/August validation/September test fixed while adding older history, three–ten training months. Balanced direct/hierarchical four configurations each, identical quarter-MAE loss without auxiliary loss; ridge/interpolation controls. Validation days1–15 select, remaining days calibrate intervals. EXPANDING_PROTOCOL.md.

Nightwatch47a057b83338d7af7dfb5ceb, worker2039, heartbeats/Telegram; log /root/.local/share/nightwatch-hackaton/47a057b83338d7af7dfb5ceb.log. GPU preflight, two-fold smoke/replays/plots passed; full process2094 active. Outputs results/expanding_vast: CSV, overall/per-zone PNG/SVG, hierarchy-benefit/stage plots. ETA00:08 Zurich October17. Next verify terminal state, retrieve/inspect graphs. More history need not improve monotonically. Hierarchical outputs are interpretable; neural input mapping remains opaque.
