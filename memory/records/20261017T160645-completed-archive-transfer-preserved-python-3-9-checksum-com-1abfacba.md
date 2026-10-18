---
title: Completed archive transfer preserved; Python 3.9 checksum compatibility repaired
  and resumed
status: historical
kind: repair
recorded_at: '2026-10-17T16:06:45.299512+00:00'
sources:
- campaigns/minimize_expanding_20261017/finalize_local.py
- results/minimize_expanding_20261017/completion.json
experiment_id: minimize_expanding_20261017
supersedes: []
status_updated_at: '2026-10-17T16:07:52.733801+00:00'
---

Verified 17October2026 around18:06Zurich: original local backup Nightwatch8c78bfcc73e7ff61f0b958cd is authoritatively crashed exit1 at18:04:59Zurich, not an observation timeout. All2428656306archivebytes had already arrived in results/minimize_expanding_20261017_completed.tar.gz.partial. Failure was AttributeError: Mac Python lacks hashlib.file_digest (Python3.9), not a computed checksum mismatch or scientific failure. Worker log /Users/miguelnogales/Documents/Codex/.nightwatch/workers/8c78bfcc73e7ff61f0b958cd.log preserved; actual failure reporting worked. Scientific Nightwatch0c174584a3b9236d9f21ffd4 remains completedexit0 and full quality result0.708892 unchanged.

Repaired digest() to portable8MiB streaming hashlib.sha256 updates. Added partial-file reuse: hash existing full-length partial first, skip downloading when hash matches; only verified bytes/SHA permit atomic rename. No fitted model, raw prediction, selection or scientific training source changed. Started new durable Nightwatcha6689a6ede6fcf6102ea8476 with3minute estimate, worker13491, log/Users/miguelnogales/Documents/Codex/.nightwatch/workers/a6689a6ede6fcf6102ea8476.log; heartbeat_expectedtrue,180secondtimeout. Actual launch18:06:24Zurich. Execution permissions preserved via approved escalation. Existing full transfer is retained; no blind restart or deleted evidence.

Next confirm the repaired job's real completedexit0 and backup_verification.json (2428656306bytes,SHA2566263d9223b7bcb7c9b4855ced55a74e4844ea1463d175866a5072d08236701d0), then update completion audit/interpretation/project summary and mark goal achieved. Do not claim checksum success before actual receipt.
