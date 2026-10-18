# HACKATON project memory

Objective (2026-10-17): compare all simple architecture candidates with expanding windows and conformal intervals. Official data only, four zones, hourly means within 0.0001 EUR/MWh; no external/pretrained/October prices. Strictly prior observed history is allowed. All outcomes are developmental.

Active: root@93.91.156.85:55340, /workspace/HACKATON; RTX5090 UUID GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9. Nightwatch034d5133ebc5e076af6aed88, worker2088/runner2089; log /root/.local/share/nightwatch-hackaton/034d5133ebc5e076af6aed88.log. Twenty-minute initial estimate ends10:57 Zurich. EXPANDING_SIMPLE_PROTOCOL.md; results/expanding_simple_5090. Eight expanding February–September tests plus eight fixed-September history-length folds;224 candidates/fold,8 workers, ridge/interpolation controls. Next-month days1–15 select, remainder calibrate; following month tests. Smoke2fold/20candidate independent audit passed; full run started, completion unverified. Report50/90/95% marginal/hour/day coverage;95% day bands infinite with at most16 calibration dates. Dependence/drift prevents unconditional nominal guarantees.

Prior224-model September search completed/audited: results/simple_5090. Linear1170parameters/MAE5.209794; CNN40579/5.222494 won August selection. Prior ensemble5.091209 remains stronger. No deployment replacement.

Earlier context preserved in memory/records/20261017T083747-startup-context-before-conformal-expanding-search-0d8ba809.md. Old expanding artifacts results/expanding_vast have16folds/64scores, remote root@ssh1.vast.ai:24199, Nightwatch47a057b83338d7af7dfb5ceb status unchecked. Ando all363 refit /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1, Nightwatch816b1393f18c891ba8a8de11 status unchecked; DEPLOYMENT_REFIT.md. Hierarchy benefit unproven; INTERPRETABLE_PROTOCOL.md.

Next: verify full terminal/audit status, retrieve artifacts, inspect overall/per-zone learning curves and conformal coverage/widths.
