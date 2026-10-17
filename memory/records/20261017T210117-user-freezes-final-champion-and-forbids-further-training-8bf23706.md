---
title: User freezes final champion and forbids further training
status: current
kind: decision
recorded_at: '2026-10-17T21:01:17.107790+00:00'
sources:
- CHAMPION_FREEZE.json
- AGENTS.md
- BEST_MODEL.md
- results/submission_champion_20261017/champion_submission/MANIFEST.json
- campaigns/monthly_ratio_20261017/sources/tune_parallel_cnn.py
experiment_id: null
supersedes: []
---

On 17 October 2026 user explicitly selected the latest champion discussed and instructed: 'freeze everything, not any more training ever'. Final artifact champion_348fit_15cal_20261017 has two MLPs, two temporal CNNs and two analog 2D CNNs, each 1/6. Verified all 17 MANIFEST.json hashes locally without modifying the bundle. CHAMPION_FREEZE.json pins model, manifest and archive SHA256; AGENTS.md, README.md, BEST_MODEL.md and best_model.json record freeze. No further training/refits/tuning/recalibration/feature or weight changes authorized. Read-only SSH process check on root@93.91.156.85:55340 found no matching HACKATON/run.py/refit.py/train.py/trainq.py Python process. Not a census of other hosts. Prepared campaigns/champion_sensitivity_20261017/lag_importance.py, syntax checked only; not uploaded/launched; no Nightwatch job created. Earlier feature-importance report covers old models, not current champion. The ridge/forest/boosting/node-attention quote is from historical submission/README.md, which was preserved unchanged. Analog retrieval is per zone/hour using five-hour relative curve MSE + 0.1 times squared center-price difference, previous 90 calendar days, up to ten neighbors weighted 1/(25+distance), then normalize and average prior interpolation residuals. Historical research scores remain eight-fold developmental evidence, not final-refit held-out scores. Previous startup summary preserved verbatim in memory/records/20261017_startup_before_final_champion_freeze.md.
