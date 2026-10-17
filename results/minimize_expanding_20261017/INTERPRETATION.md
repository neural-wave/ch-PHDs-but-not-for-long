# Expanding-training minimization: verified result

The prior-data selector improves the primary equal-month MAE ratio from
**0.721285 to 0.708892**, a **1.718% relative reduction**. It has **29.111% less
MAE than mean-corrected linear interpolation** under the eight-month metric.
Pooled MAE falls from **4.181172 to 4.107397 EUR/MWh**. Against the later
zone-weight pre-test policy (0.719952), the new selector improves the ratio by
**1.536%**. These comparisons use the same February–September targets.

The distinct pooled model/corrected-interpolation MAE ratio is 0.706367.
For the explicitly alternate, uncorrected linear-interpolation denominator,
the equal-month ratio is 0.672772 and the pooled ratio is 0.669852. The primary
score remains 0.708892 against **mean-corrected** interpolation. These
statistics use ratios after aggregating absolute errors, rather than averaging
quarter, daily or zone ratios. Full tables: aggregate_comparison.csv and
alternate_interpolation_monthly_scores.csv.

| Test month | Training months | Training days | Original selector ratio | New selector ratio |
|---|---:|---:|---:|---:|
| February | 3 | 91 | 0.762760 | 0.759890 |
| March | 4 | 122 | 0.712257 | 0.687653 |
| April | 5 | 150 | 0.777091 | 0.736378 |
| May | 6 | 180 | 0.726375 | 0.708228 |
| June | 7 | 210 | 0.663480 | 0.666538 |
| July | 8 | 241 | 0.702687 | 0.695008 |
| August | 9 | 271 | 0.703328 | 0.696703 |
| September | 10 | 302 | 0.722305 | 0.720740 |

Seven months improve; June worsens by 0.46% in raw MAE. The largest gains
are March and April. All model ratios remain below one. The worst new
monthly ratio is 0.759890. Training always begins in October 2025 and expands
by one month per fold. Calendar month and training size change together, so
these results do not isolate the causal effect of additional training data.

The lowest reported aggregate recipe is `fixed_blend_ratio`: ratio 0.707968,
pooled MAE 4.103184. This is a descriptive grid winner. The value 0.708892
belongs to the separate rule that selects its recipe using preceding-month
data before each test. That rule chooses `combined_blend_uniform` for seven
folds and `combined_blend_ratio` for July. The aggregate grid winner is not
substituted for this pre-test selector.

This campaign jointly changed preprocessing, allowed equal-seed means, and
expanded the fixed blend menu. Its overall gain cannot be attributed to
normalization alone. It adds 384 fixed-physical-scale fits and reuses the
audited standard-scale controls. Features retain engineered physical units,
fixed clipping at 15, and fitting-only additive knots. Models, seeds,
ordinary/ratio training objectives and budgets are frozen in the protocol.
Every fold starts fitting from scratch. All selection uses the preceding
month's first 15 days; calibration uses its separate remaining days.

The selector retains all eleven ordered quantiles. Its calibrated intervals
have the following pooled empirical coverage and mean widths:

| Nominal interval | Coverage | Width, EUR/MWh |
|---|---:|---:|
| 50% | 53.28% | 5.8959 |
| 80% | 81.31% | 13.0791 |
| 90% | 90.71% | 18.8409 |
| 95% | 95.35% | 25.0793 |

Narrow/middle/wide relative-width groups have actual MAE 2.1667/3.5583/6.4727,
against same-group interpolation MAE 2.4975/4.7387/10.0847. Width indicates
relative uncertainty, not a probability of correctness. Coverage is empirical
under temporal dependence. The conserved q50 is a constrained decision;
nonmedian marginal quantiles do not have an hourly-mean constraint.

The independent audit recomputed 384 candidate selection scores, replayed
281 selected new/control checkpoints with zero discrepancy, verified 92,544
exported quarters and found maximum hourly-mean error 1.14e-13 EUR/MWh. It
checks chronology, source hashes, training-only weights/knots, fixed
preprocessing, seed averages, blend menus, selection argmins, quantile order,
CQR ranks/nesting, confidence labels and aggregate metrics. The separate
comparison script recomputes both selectors' monthly MAE from matched saved
quarter arrays and independently reconstructs interpolation.

These are developmental backtests, with one chronological inner selection
block. They are not a blind benchmark or independent-fold significance test.
The challenge production model and submission were not replaced.

Evidence: REPORT.md, audit.json, comparison.json,
verified_selector_comparison.json, selector_comparison_by_training_size.csv,
selector_interval_comparison.csv, monthly_zone_scores.csv, interval_metrics.csv,
confidence_bins.csv and expanding_selector_comparison.png/.svg. Protocol and
reproduction sources: campaigns/minimize_expanding_20261017/. Source/checkpoint
recovery archive: results/minimize_expanding_20261017_completed.tar.gz;
remote SHA256 6263d9223b7bcb7c9b4855ced55a74e4844ea1463d175866a5072d08236701d0.

Scientific Nightwatch 0c174584a3b9236d9f21ffd4 completed with exit 0 at
17:36:04 Zurich on 17 October 2026. Host root@93.91.156.85:55340,
/workspace/HACKATON; log
/root/.local/share/nightwatch-hackaton/0c174584a3b9236d9f21ffd4.log.
Local backup is now checksum-verified: 2,428,656,306 bytes with the same SHA256
as the remote completion receipt. backup_verification.json also records local
metadata hashes. The original backup job 8c78bfcc73e7ff61f0b958cd transferred
all bytes, then crashed because Python 3.9 lacks hashlib.file_digest. Portable
streaming SHA256 repaired that verifier and reused the existing transfer.
Repaired Nightwatch a6689a6ede6fcf6102ea8476 completed with exit 0 at
18:06:39 Zurich. Its worker log is
/Users/miguelnogales/Documents/Codex/.nightwatch/workers/a6689a6ede6fcf6102ea8476.log.
