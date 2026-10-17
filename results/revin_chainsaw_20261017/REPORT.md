# Predicted shape versus quantile dispersion

Requested diagnostic: bands should spread for predicted chainsaw quarter-hour shapes and narrow for predicted smooth shapes. This analysis tests that behavior; it does not impose it on the model.

Roughness is mean absolute second difference of the four q50 prices within each hour, divided by input-only daily scale sqrt(hourly variance+1). It is zero for a linear segment. Smooth/chainsaw are descriptive lower/upper-quartile roughness labels fixed from preceding calibration q50 predictions per zone, not a risk classifier. Shape here means predicted median; observed shape is a separate diagnostic.

                                        hours  predicted_roughness  observed_roughness  normalized_width90  normalized_raw_width90  normalized_mae
recipe                 predicted_shape                                                                                                            
compact_day_reversible chainsaw          5993             0.160640            0.348978            0.725111                0.682761        0.188309
                       middle           11495             0.063393            0.213160            0.617315                0.576804        0.122100
                       smooth            5648             0.020176            0.132957            0.536884                0.498337        0.078589
compact_fixed          chainsaw          5875             0.172445            0.349934            0.827131                0.751037        0.192212
                       middle           11208             0.064630            0.218717            0.579761                0.511514        0.121647
                       smooth            6053             0.020600            0.129754            0.425049                0.365359        0.075622

{
  "compact_day_reversible": {
    "mean_within_day_rank_correlation": 0.47268085874075405,
    "fraction_positive": 0.9688796680497925,
    "valid_day_zones": 964
  },
  "compact_fixed": {
    "mean_within_day_rank_correlation": 0.4983510734259426,
    "fraction_positive": 0.9854771784232366,
    "valid_day_zones": 964
  }
}

within_day_associations.csv measures roughness/width association across24hours with identical day/zone scale, directly addressing scale confounding. Interpret alongside raw90 widths, calibration effects and observed error. The example panels pick the day/zone with greatest predicted roughness contrast, then its smoothest and most jagged hours. No observed error selects examples. These are exploratory extrema in developmental folds.

Marginal quantile bands measure dispersion of each quarter, not joint trajectory oscillation or unfamiliarity. A confidently predicted chainsaw can have narrow bands; a smooth median can hide uncertain alternating trajectories. Learned-affine RevIN changes scaling and representation but does not guarantee a shape-width relationship. No such mechanism is claimed from the plots alone.
