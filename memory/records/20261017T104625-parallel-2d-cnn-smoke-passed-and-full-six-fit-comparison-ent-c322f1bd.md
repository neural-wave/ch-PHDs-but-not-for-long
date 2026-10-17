---
title: Parallel 2D CNN smoke passed and full six-fit comparison entered
status: historical
kind: verification
recorded_at: '2026-10-17T10:46:25.847970+00:00'
sources:
- results/parallel_2d_zone_cnn_20261017/smoke.json
- PARALLEL_2D_CNN_PROTOCOL.md
experiment_id: parallel_2d_zone_cnn_20261017
supersedes: []
status_updated_at: '2026-10-17T11:27:41.627673+00:00'
---

2026-10-17 smoke passed=true on VastRTX5090. Exact pairedQ/H rows and hourly24repeat4 values checked; query-dayQ zero/mask0, queryH mask1, missing prior days zero/masked. Features invariant to current/future quarter perturbation and past-only reconstruction. Cross-zone gradient and zero hourly residual mean passed; convweightshape8x5x2x8 checked. Six shorttrialsmoke passed. Full search entered at2workers; log WORKERS2, Nightwatch1b9cf34ee35a62b780645756 durable wrapper5024. Local smoke artifact copied. Final numerical/independent audit/filterplot export still pending; ETA12:51Zurich initial conservative.
