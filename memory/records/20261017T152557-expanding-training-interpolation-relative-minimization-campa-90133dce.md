---
title: Expanding-training interpolation-relative minimization campaign implemented
  and launched
status: historical
kind: experiment
recorded_at: '2026-10-17T15:25:57.859623+00:00'
sources:
- campaigns/minimize_expanding_20261017/PROTOCOL.md
- campaigns/minimize_expanding_20261017/train.py
- campaigns/minimize_expanding_20261017/audit.py
- campaigns/minimize_expanding_20261017/audit_selection.py
- campaigns/minimize_expanding_20261017/run.py
- results/minimize_expanding_20261017_launch.json
- memory/records/20261017_startup_before_minimize_expanding.md
experiment_id: minimize_expanding_20261017
supersedes: []
status_updated_at: '2026-10-17T15:27:02.762471+00:00'
---

User explicitly requests minimizing the new error ratio against linear interpolation, and emphasizes an increasingly wide training set. Preserve exact VALIDATION_PROTOCOL.md eight expanding February–September tests: training October–December2025 first, adding one month per fold through October2025–July2026; preceding-month first15days select,16–end calibrate. No rolling-window shortening/refit. Primary equal-month q50 MAE/mean-corrected-linear-interpolation MAE; lower is better. Quantiles always retained, official-only and strictly prior history, hourly conservation; developmental results.

Implemented frozen campaign campaigns/minimize_expanding_20261017:384new fixed-physical-scale fits(six strongest prior families ×ordinary/ratio objectives×two learningrates×seeds17/43×eight folds). Same prior architectures, MAE+eleven-quantilepinball,300/35epochs/patience(analogCNN450/50). Fixed engineered units with clip15; additive knots fitting-only. Validation compares singles and equal-seed means at each LR, new fixed blends and combined fixed/standard blends. Menu:rich-linear pairs.25/.5/.75,equaltop2/3/4/6, original/fixed blend fallbacks. Frozen audited standard controls reused; no extra retraining or outer-label selection. Final pre-test selector distinct from aggregate test winner. All raw quantiles/CQR50/80/90/95, confidence diagnostics and source/checkpoint archive retained. Independent audit reconstructs candidates/menus/calibration and selected new+standard checkpoint replays.

Existing authorized execution host root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python,RTX5090. Earlier S4D Nightwatchd85b1d1b1b58a45af40c3a57 independently observed completedexit0 at15:24:39UTC; completionreceipt archive333149705bytes,SHA256f1910a8204d197012990c76a5edab76e080afeac12fded998ea016791efd59e8,audit_passedtrue. It no longer occupies GPU. This record does not inspect its scientific scores or establish local backup.

First new Nightwatche69e66e5c16d69c205bdb868 failed during prepare before any fit: pandas rejected numpy.str_ in reused feature causality probes. Fixed by converting date arrays to Python-str object arrays; original failed receipt/log preserved. Corrected Nightwatch0c174584a3b9236d9f21ffd4 launched17:25:38Zurich with20minute estimate,worker11867,log/root/.local/share/nightwatch-hackaton/0c174584a3b9236d9f21ffd4.log. Current wrapper source hash matches installed414d11592b7a5edd46e90fa913dd79fad4dbc6826975327aca4c17b8702343f5. heartbeat_expectedtrue,timeout180seconds; tracker health and authorized owner Telegram enabled verified. Full success requires prepare, first/lastfold smoke+audit,384fullfits+audit,comparisonreport/recoveryarchive. At registration full results remain pending; verify subsequent heartbeat, preflight and actual training before final status. Prior baseline originalblend0.720494,originalpretestselector0.721285, posthoczonebest0.718907/pretestzonepolicy0.719952. Do not claim this run improved any metric until completed audited evidence.

Prior startup preserved verbatim in memory/records/20261017_startup_before_minimize_expanding.md. Next: verify running full-training stage, then on completion retrieve and audit reports and SHA256 archive; compare new selector with original selector, report every month/zone and uncertainty. No deployment replacement.
