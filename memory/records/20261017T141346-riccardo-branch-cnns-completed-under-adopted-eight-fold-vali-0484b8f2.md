---
title: Riccardo branch CNNs completed under adopted eight-fold validation; existing
  linear wins
status: current
kind: comparison
recorded_at: '2026-10-17T14:13:46.674825+00:00'
sources:
- VALIDATION_PROTOCOL.md
- campaigns/riccardo_comparison_20261017/PROTOCOL.md
- campaigns/riccardo_comparison_20261017/upstream_receipt.json
- campaigns/riccardo_comparison_20261017/completion.json
- campaigns/riccardo_comparison_20261017/retrieval_receipt.json
- results/riccardo_comparison_20261017/REPORT.md
- results/riccardo_comparison_20261017/independent_audit.json
- results/riccardo_comparison_20261017/conclusions.json
- compare_riccardo.py
- audit_riccardo_comparison.py
- summarize_riccardo_comparison.py
experiment_id: riccardo_comparison_20261017
supersedes: []
---

Requested 17 October 2026: inspect https://github.com/neural-wave/ch-PHDs-but-not-for-long branch riccardo and compare its best method under our validation against completed methods.

Pinned commit 1ed1423c5fa05586457347ab34e69b9bcfb23031 (17 October 15:50:40 CEST); Git authentication succeeded. Official training CSV SHA256 equals our verified f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c. Vendored only experiment sources/metrics/checkpoints in campaigns/riccardo_comparison_20261017/upstream. External identification and October data excluded from campaign inputs.

Lowest committed August–September aggregate MAE: reduced-profile CNN with revised weights 5.087239411; selected reduced features 5.087528729; README raw-hourly model 5.113040454. Original selection uses August and September together; all-year 11-month LOMO scorers leak their labels into training-month weights even with train-month-only minmax. These original selection figures cannot be compared directly with our held-out or eight-month score.

Implemented fresh fits of both variants under VALIDATION_PROTOCOL.md: eight expanding train blocks October-start 3–10 months; next-month first15 selection; days16-end reserved/unused; following full-month test; fresh train-only LOMO month scores, minmax/sample weighting and scalers; imported per-zone configs/seeds fixed, 150epochs/patience16/AdamW/QweightedMAE1.5,1,1,1.5. Four-channel reduced profiles use hour and weekday embeddings; raw-hourly CNN has one unnormalized input and no calendar. Internal LOMO retains held-month early stopping and may train on later months inside the outer training block. Historical configs were selected using inspected Aug/Sep; developmental adaptation, not blind/full nested selection or controlled architecture ablation. Existing methods can use causal history/cross-zone data and have different search budgets.

Completed on existing authorized root@93.91.156.85:55340, /workspace/HACKATON, RTX5090 GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9; /venv/main/bin/python, Python3.12.15, NumPy2.5.4, Torch2.14.1+cu130/CUDA13.0, TF32 disabled. Upstream pins NumPy2.5.3/Torch2.8.0 and reports MPS: runtime optimization numerics may differ. No local Mac training.

Nightwatch 74e4190bed4235d2126ac27b confirmed completed exit0 at 2026-10-17 16:02:45 CEST; started16:00:16, worker8132, log /root/.local/share/nightwatch-hackaton/74e4190bed4235d2126ac27b.log. Heartbeat expected/180s timeout. No active comparison job. Full run:208 inner scores,64 final models; independent audit32zonefolds/64replays, official target/split/scale/weight/selection/MAE/baseline audits passed; maximum replay difference0, mean conservation1.1369e-13. Smoke adapter parity and cal/test-target perturbation invariance passed.

Primary mean monthly ratio / pooled MAE:
our linear .738302973 /4.273507576;
linear quantile .739332896 /4.278685035;
additive .753149720 /4.366023432;
ridge .754628111 /4.373205069;
hierarchy .762694960 /4.419718506;
direct .765103932 /4.433215471;
CNN .773003024 /4.488442378;
MLP .776817326 /4.476401111;
Riccardo reduced .796833773 /4.615017205;
Riccardo raw .803239319 /4.658694671;
interpolation1 /5.814817121.
Imported reduced ranks9/11 and raw10/11. Both lose to our linear every month and beat interpolation every month. Reduced has7.9277% higher primary ratio and7.9913% higher pooled MAE than linear. Same241days/92,544targets.

September alone: reduced5.543063345, raw5.597739868; our linear5.200312773, analog90CNN5.177024160, jointblend5.091208820. Joint blend uses fullAugust, unlike first15 selectors; September-only methods have no matched eight-month estimates. Separate table in september_comparison.csv.

Full result/checkpoint/prediction trees retrieved locally. Base completed archive results/riccardo_comparison_20261017_completed.tar.gz matches remote SHA25626bce09651a8ec3a88f974ef3835bb17eedfc0258e56c1e7437c13a6d7776a4d. Final reporting/September appendix added afterward via summarize_riccardo_comparison.py. Monthly plot visually inspected. No deployment changes. Next: retain existing pipeline; any new feature/architecture tuning must use adopted scoring and preserve exploratory/eligible distinction.

Verbatim prior startup summary preserved before shortening:
# HACKATON project memory

Objective: minimize average monthly MAE / interpolation_MAE for official four-zone quarter reconstruction. Preserve hourly means within 0.0001 EUR/MWh, strictly prior observed quarter history, official-only inputs and frozen inference parameters. No external/pretrained components or October evaluation prices in submissions. September remains developmental; zones anonymized; user prefers small interpretable CNNs.

Protocol: VALIDATION_PROTOCOL.md. Eight expanding February–September tests, three initial training months; next-month days1–15 selection, remaining days reserved for calibration. Train-only normalization/fitting. Primary score averages same-month model MAE / mean-corrected-linear-interpolation MAE equally. Fixed-September history sizes stay separate.

Active 2026-10-17: Nightwatch b6d61339b6b5b82131b36c0d, 384 neural fits across six families, ordinary versus monthly-ratio training; ridge/analog/blend controls. Relative loss weights use only fitting-month baseline errors and day counts. All learned models receive today's complete hourly curve. Source snapshot isolated in campaigns/monthly_ratio_20261017/sources/. Smoke audited: 44 candidates, 24 exact checkpoint replays, mean error≤1.14e-13. Full results pending. Host root@93.91.156.85:55340,/workspace/HACKATON; /venv/main/bin/python. Worker7626; log /root/.local/share/nightwatch-hackaton/b6d61339b6b5b82131b36c0d.log. Launched15:57Zurich; revised estimate16:04 after4/8folds. Heartbeat/Telegram enabled; success requires full audit/archive.

Prior benchmark: linear ratio0.738303/pooledMAE4.273508; linearquantile0.739333. September-only jointblend5.091209, analog90CNN5.177024, quantilecohort5.216651 lack matched monthly evidence. Offline363-day refit audited/no submission. Exploratory FlowState completed with no incremental gain; verified archive results/flowstate_20261017_completed.tar.gz.

Evidence: results/monthly_ratio_20261017/{launch.json,manifest.json,smoke/audit.json}; campaign PROTOCOL.md. Detail and verbatim prior startup: memory/records/20261017T135818-monthly-ratio-minimization-campaign-launched-384-paired-neur-d159051c.md. Next verify actual terminal state, full REPORT.md/audit.json, then retrieve recovery archive and hash; no deployment replacement.
