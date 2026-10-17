---
title: Startup context before conformal expanding search
status: historical
kind: continuity
recorded_at: '2026-10-17T08:37:47.756189+00:00'
sources:
- project_memory.md
experiment_id: null
supersedes: []
---

# HACKATON project memory

Objective: official-only quarter-price reconstruction for four zones; preserve hourly means within 0.0001 EUR/MWh. No external/pretrained/October prices. Fit October2025–July2026, select August, development-test September; causal prior observed quarter history allowed. September has already been inspected.

Verified 2026-10-17: new RTX5090 search completed224 single models using8 parallel workers (short benchmark4/8/12:8.06/6.15/6.84seconds; GPU99% observed). August-selected family representatives: linear1170parameters SeptemberMAE5.209794; additive4671/5.255107; MLP13635/5.343947; CNN40579/5.222494. CNN won overall August selection. Independent224-score/selection/conservation audit passed; max mean error5.68e-14. SIMPLE_SEARCH_PROTOCOL.md; results/simple_5090/{metrics.json,independent_audit.json,scores.csv,simplicity.png}. Prior ensemble5.091209 remains stronger; prior singleMLP5.234445. No deployment replacement.

New host root@93.91.156.85:55340, /workspace/HACKATON, GPU UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9. Nightwatch2093a7ebed349785229d04bf finished exit0, notification delivered; log /root/.local/share/nightwatch-hackaton/2093a7ebed349785229d04bf.log. No active job from this search.

Earlier jobs not rechecked: Ando all363 refit Nightwatch816b1393f18c891ba8a8de11, /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1; DEPLOYMENT_REFIT.md. Expanding artifacts verified2026-10-17: results/expanding_vast/{learning_curves.csv,metrics.json},16 folds/64 scores (8 expanding Feb–Sep,8 fixedSep); audits reported passed. Starts3 trainingmonths, not1. Remote root@ssh1.vast.ai:24199,/workspace/HACKATON; Nightwatch47a057b83338d7af7dfb5ceb live status unchecked; EXPANDING_PROTOCOL.md. Hierarchy benefit previously unproven; interpretable additive5.688563 (INTERPRETABLE_PROTOCOL.md).

Detailed prior startup preserved in memory/records/20261017T082550-startup-summary-archived-before-new-5090-simple-search-ec074e37.md. Next: retain these single-model artifacts; verify older runs separately before interpreting their status.
