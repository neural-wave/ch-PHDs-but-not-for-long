# Linear compact with RevIN

Fresh RevIN fits of the15exact source-selected linear-compact checkpoints across eight expanding monthly folds. Same architectures, lr/seeds, physical-currency ratio-weighted MAE+pinball objective and frozen single/seed-mean weights. Stopping epochs reselected using prior days1–15; intervals calibrated on16–end. This is a frozen-recipe transfer ablation, not a new full candidate search or a production fit.

                 recipe  folds  mean_monthly_ratio  pooled_mae  mean_monthly_pinball_ratio  coverage90   width90  worst_ratio
original_linear_compact      8            0.724553    4.201780                    0.215858    0.890052 18.237317     0.766425
   linear_compact_revin      8            0.733498    4.254934                    0.220878    0.924079 23.624012     0.788654
    validation_selector      8            0.733498    4.254934                    0.220878    0.924079 23.624012     0.788654
          interpolation      8            1.000000    5.814817                    0.309732    0.895023 27.783635     1.000000

Original linear compact ratio must recover0.7245525295146528. Lower is better. All months are developmental. No deployment or registry promotion.

Full-day panels compare original linear compact(left) with its fresh RevIN refit(right). full_day_normalized.png/svg has shared normalized axes; full_day_eur_mwh.png/svg shows original currency. Same March14ZONE2 and April26ZONE3 examples as the previous plot, fixed before this run. Raw quantile lines and calibrated90%bands are distinct. Examples are illustrative; no observed outcome was used to select new model weights.

      date  zone                  recipe   width90       mae  coverage90  normalized_width90  normalized_mae
2026-03-14     2 original_linear_compact  6.469110  3.329027    0.625000            0.457715        0.235542
2026-03-14     2    linear_compact_revin  9.916531  3.313838    0.770833            0.701633        0.234467
2026-04-26     3 original_linear_compact 40.385059  9.493595    0.822917            0.288448        0.067807
2026-04-26     3    linear_compact_revin 90.967346 11.133440    0.989583            0.649729        0.079520

Source protocol/frozen_components.json and audit.json document input-only scales, learned-affine residual reversal, causality, exact weights, checkpoint replay, ordered quantiles, hourly conservation and CQR validation. Full predictions/checkpoints retained remotely in the recovery archive.
