# Scale-adjusted uncertainty comparison

Both models were trained fresh; RevIN reverses input-only daily scale and its learned positive affine on output deviations. Quantile plots use shared daily-normalized axes. Example dates were previously inspected, so do not treat them as independent confirmation.

Normalized narrow/middle/wide thresholds come from preceding calibration outputs within zone/quarter. Pooled values are target-weighted diagnostic quantities, separate from the primary equal-month MAE/interpolation score. Scaling errors inversely by daily volatility here does not change the original-currency training loss.

                                   n  normalized_mae  normalized_width90  coverage90  normalized_error_ratio
recipe                 label                                                                                
compact_day_reversible middle  45130        0.120670            0.599540    0.912320                0.806710
                       narrow  22624        0.079379            0.459667    0.932240                0.944340
                       wide    24790        0.188063            0.824486    0.895522                0.720224
compact_fixed          middle  38646        0.121151            0.558835    0.900844                0.795047
                       narrow  29240        0.083136            0.344182    0.873769                0.836719
                       wide    24658        0.190150            0.975752    0.918647                0.740566

within_scale_associations.csv gives rank association within fold, zone and low/middle/high daily-scale groups defined from calibration quartiles. These are descriptive dependent-data diagnostics; no test thresholds or models are selected. Check consistency across these strata rather than interpreting pooled association alone. No novelty/rarity mechanism or deployment promotion is established.

See REPORT.md, audit.json, selected_affine.json, day_uncertainty.csv, normalized_confidence_monthly.csv, normalized_confidence_pooled.csv and uncertainty_examples.png/svg.
