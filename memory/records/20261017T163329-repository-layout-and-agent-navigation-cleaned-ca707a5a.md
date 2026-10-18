---
title: Repository layout and agent navigation cleaned
status: current
kind: experiment
recorded_at: '2026-10-17T16:33:29.782345+00:00'
sources:
- README.md
- AGENTS.md
- run.py
- docs/layout_migration.json
- docs/CODE_INDEX.md
- results/model_ranking_20261017/audit.json
experiment_id: null
supersedes: []
---

2026-10-17: Moved 108 root Python files to code/, operating guides to docs/, older protocols to docs/protocols/, original challenge PDF/readme to docs/reference/. Complete old-to-new mapping: docs/layout_migration.json. Added root README.md, AGENTS.md, run.py launcher, docs/README.md and docs/CODE_INDEX.md. Launcher sets repository cwd, shared module paths and inherited PYTHONPATH; sibling subprocess paths and source-root detection updated. predict_best imports deferred until after argument parsing so --help works without ML dependencies. Historical campaign and frozen submission files not edited. Some old campaigns require original root source paths/hashes; documented that limitation rather than altering provenance. Large artifacts remain excluded from Git. Metadata rank regeneration passed across 10 campaigns and 110 primary recipes; champion registry values unchanged except current validation document checksum. Verified AST syntax for 109 files, all navigation links, migration map destinations and all frozen submission manifest hashes including local model. Full inference not run: system Python lacks pandas, bundled Python lacks torch; no training run locally. Prior startup preserved verbatim in memory/records/20261017_startup_before_layout_cleanup.md; startup shortened. Commit/push to existing authorized el_bicho branch follows.
