---
title: 'Normalization completed: fixed physical scaling wins across compact CNN S4D
  S5'
status: current
kind: result
recorded_at: '2026-10-17T15:18:10.157817+00:00'
sources:
- results/normalization_20261017/REPORT.md
- results/normalization_20261017/audit.json
- results/normalization_20261017/metrics.json
- results/normalization_20261017/insights.json
- results/normalization_20261017/paired_candidate_ratios.csv
- results/normalization_20261017/paired_selected_ratios.csv
- results/normalization_20261017/completion.json
- results/normalization_20261017/launch.json
- results/normalization_20261017/backup_launch.json
- campaigns/normalization_20261017/PROTOCOL.md
experiment_id: normalization_20261017
supersedes:
- memory/records/20261017T150451-normalization-audit-and-matched-four-arm-eight-month-quantil-4cf6c4ca.md
---

Verified actual Nightwatch/owning artifacts17October2026: scientific run73994759a86ac27184894e2a completedexit0 at17:11:31Zurich; all8expanding February–September folds and384fits done. Remote host root@93.91.156.85:55340,/workspace/HACKATON; worker log/root/.local/share/nightwatch-hackaton/73994759a86ac27184894e2a.log. Independent audit384checkpoints passed, maximum replay error0; parameter counts matched within family (compact19180,S4D31020,S531036), independent normalization equations/heldout preprocessing invariance/flat-negative inputs/physical loss equivalence/chronology/CQR ranks/quantile order/hourly conservation/aggregate metrics all passed. Scientific result complete; backup finalization separately tracked below.

Primary metric is arithmetic mean of8monthly conserved-q50MAE/mean-corrected interpolation MAE ratios. Arm ratios (fixed,standard,day_input,day_reversible): compact(.769237824,.797334266,.807164687,.812202282);S4D(.763496712,.782517606,.796951354,.802287378);S5(.762926204,.789863707,.799327307,.805933348). Fixed uses engineered /100level,/20difference/residual units and common[-15,15]clip, not literal unscaled prices. Standard adds fitting-set z-scores. Daily arms use only query hourly curves, retain absolute level/std side information, and are reversible-normalization adaptations without learned affine parameters. Every arm uses the same physical-currency MAE+pinball loss and same matched rates/seeds/budget. No inference about an exact original RevIN implementation.

Relative reduction of fixed versus validation-selected standard arm: compact3.5238%,S4D2.4307%,S53.4104%. Fixed wins7/8monthly selected comparisons for every family; across all96matched month/family/lr/seed pairs it wins88 (compact30/32,S4D30/32,S528/32). Mean paired ratio differences fixed minus standard:-.031447compact,-.024915S4D,-.026624S5; mean differences daily input/reversible versus standard positive for all three. Fixed improves for both seeds in every family, not a single lucky selected seed. These are overlapping developmental backtests and descriptive matched evidence, not96independent significance samples. Primary allconfig relative reductions3.9207%,3.1761%,3.3738%, respectively. Local insights.json records exact calculations; independently recomputed equal-month/pooled aggregates from tiny saved score CSVs.

Fixed also improves normalized pinball by4.2626%compact,3.5102%S4D,4.6226%S5 versus standard. Daily reversible calibrated90%interval widths increase11.5125%compact,23.7144%S4D,10.5785%S5 versus standard, without better point accuracy. Best descriptive recipe S5fixed: meanmonthlyratio.762926204,pooledMAE4.422713EUR/MWh,90%coverage89.3683%,width19.40135EUR/MWh; S4Dfixed.763496712/4.430358 and compactfixed.769237824/4.458661 are close. S5 gains only.8205%relative ratio over compactCNN, too small here to claim decisive architectural superiority. Preceding-block validation selector ratio.774027622,pooledMAE4.490647,90%coverage90.0836%. Best observed outer aggregate recipe is descriptive; selector is distinct.

Interpretation: extra z-score/daily normalization did not help under this matched budget. Existing physical units appear better suited to these engineered features; causal explanation is unproven. No general claim that normalization is universally harmful. Existing stronger quantile ensemble remains about.72, so this study does not displace it. Useful follow-up, not launched: matched fixed-scale ablation on stronger rich-linear/MLP/CNN ensemble components. No deployment change.

Fetched scientific report/metrics/CSV/audit/manifests locally. Remote recovery archive results/normalization_20261017_completed.tar.gz1113362255bytes,SHA2563a8096ec7bee060296665a478c3a967ad3cc7f4e00ad0a78b2b7a28d30a688ce. Large local archive transfer still in progress at this record. Nightwatch local backup-finalization job8fdc7710e2b4dcb2e26b4158,worker8794,adopts actualscpPID8474 and reports success ONLY after process termination and size/SHA match; nonmatching/incomplete archive raises failure. ETA17:21:32Zurich; log/Users/miguelnogales/Documents/Codex/.nightwatch/workers/8fdc7710e2b4dcb2e26b4158.log; expected final receipt results/normalization_20261017/backup_verification.json. Success/failure alerts and180s heartbeats enabled. Do not claim local archive checksum verified until receipt.

Source-provenance check found the execution snapshot differs from current local interpretable_models.py (local backward-compatible optional interpolation_days extension, default0) and VALIDATION_PROTOCOL.md (local quantile-default text). This does not establish when either changed. Exact24execution source files were hash-verified remotely, packaged in execution_sources.tar.gz, retrieved and individually hash-verified/extracted under results/normalization_20261017/execution_sources/. Other local source hashes match; required quantile protocol was actually implemented and audited. Preserve these execution sources for replay. Old startup preserved in memory/records/20261017_startup_before_normalization_results.md. Next: inspect backup receipt on return; strongest-family fixed-scale follow-up only if requested.
