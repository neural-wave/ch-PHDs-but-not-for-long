---
title: RTX 5090 simple search completed and independently audited
status: current
kind: result
recorded_at: '2026-10-17T08:29:43.608113+00:00'
sources:
- SIMPLE_SEARCH_PROTOCOL.md
- simple_search.py
- audit_simple_search.py
- results/simple_5090/independent_audit.json
experiment_id: null
supersedes:
- memory/records/20261017T082633-new-rtx-5090-simple-architecture-search-launched-877693f0.md
---

Verified2026-10-17: Nightwatch2093a7ebed349785229d04bf finished exit0, delivered:true; started1792225493.6379309 ended1792225588.2389243,94.601s including preparation/smoke/full/audits. Remote /workspace/HACKATON/results/simple_5090; host root@93.91.156.85:55340; log /root/.local/share/nightwatch-hackaton/2093a7ebed349785229d04bf.log. GPU0%/2MiB after completion. No search process left active. All224 candidates completed with serialized replay0-error, finite61-day predictions and conservation. Independent audit recomputed all224 August/September scores to1e-10, verified global/family August argmin, official hash, max hourly mean error5.684341886080802e-14.

All family winners use389 rich features. Linear interpolation anchor:1170parameters, August4.4246093921, September5.2097944722 (id168). Additive interpolation:4671,4.5145522774,5.2551070014(id173). MLP two32-wide hidden layers interpolation:13635,4.3432562245,5.3439467117(id190). CNN two32-channel kernel3 layers flat:40579,4.3429247599,5.2224936622(id162). Overall August winnerCNN; linear has lower September diagnostic error but is not substituted as the global August winner. Linear coefficients operate on engineered features, not raw-feature causal effects. No ensemble/deployment replacement.

Prior cached ensemble September5.091209 remains stronger; prior singleMLP5.234445. Current mean-corrected interpolation7.0903212348. September developmental, not blind; no significance claim. Full-run GPU99% observed, short-workload8workers fastest(6.1541s vs4workers8.0606/12workers6.8449), not a full-run scaling claim. Initial plot visually inspected, CNN label needed extra horizontal margin; source corrected and plot regenerated. Full artifacts packaged /workspace/simple_5090.tar.gz(78MB) for local preservation. Earlier Ando and old Vast expanding jobs not rechecked. Short project summary updated preserving old pointers.
