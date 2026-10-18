---
title: Seven-calendar-day interpolation-error feature implemented and verified
status: current
kind: implementation
recorded_at: '2026-10-17T08:52:33.215755+00:00'
sources:
- interpolation_history.py
- interpretable_models.py
- simple_search.py
- verify_interpolation_history.py
- INTERPOLATION_HISTORY_PROTOCOL.md
- results/interpolation_history_audit_20261017.json
experiment_id: null
supersedes: []
---

2026-10-17: User requested previous interpolation errors as a time series, clarified last seven days. Implemented optional compact_features(interpolation_days=7): ordered lag1–7 per-zone/hour channels containing four signed actual-minus-mean-corrected-interpolation errors, four absolute errors, one availability mask per lag. 63 added channels (27->90); calendar gaps not compressed. SIMPLE_SEARCH_FEATURES=compact,compact7 enables matched existing 224-candidate grid; default compact/rich unchanged. Historical rich features already use interpolation errors at lags1/2/7/14 and summaries; hierarchical contiguous lag implementation uses deviations from flat hourly means, a different quantity. Verified on Ando using /home/mnogales/miniconda3/envs/TSFM_PATCH/bin/python, short CPU feature/forward checks only, no training. Exact lag values, magnitudes, current/future perturbation, past-only equivalence, missing/empty history, compact prefix and four-family mean conservation passed. Evidence results/interpolation_history_audit_20261017.json. No MAE improvement established; no full training or submission refit launched. Next: matched remote ablation using August selection, September development scoring; register long job in Nightwatch. Existing expanding remote campaign untouched; its live state not inspected in this turn.
