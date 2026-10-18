# Frozen champion ensemble with RevIN

Fresh RevIN fits of the51exact selected champion leaves across eight expanding monthly folds. Same architectures, lr/seeds, physical-currency ratio-weighted MAE+pinball objective and frozen blend weights. Stopping epochs reselected using prior days1–15; intervals calibrated on16–end. This is a frozen-recipe transfer ablation, not a new full candidate search or a production fit.

             recipe  folds  mean_monthly_ratio  pooled_mae  mean_monthly_pinball_ratio  coverage90   width90  worst_ratio
  original_champion      8            0.707968    4.103184                    0.207375    0.905872 18.835448     0.754293
     revin_champion      8            0.725133    4.206289                    0.217832    0.924598 23.610792     0.782911
validation_selector      8            0.725133    4.206289                    0.217832    0.924598 23.610792     0.782911
      interpolation      8            1.000000    5.814817                    0.309732    0.895023 27.783635     1.000000

Original champion ratio must recover0.7079682867567635. Lower is better. All months are developmental. No deployment or registry promotion.

Full-day panels compare original champion(left) with its fresh RevIN refit(right). full_day_normalized.png/svg has shared normalized axes; full_day_eur_mwh.png/svg shows original currency. Same March14ZONE2 and April26ZONE3 examples as the previous plot, fixed before this run. Raw quantile lines and calibrated90%bands are distinct. Examples are illustrative; no observed outcome was used to select new model weights.

      date  zone            recipe    width90       mae  coverage90  normalized_width90  normalized_mae
2026-03-14     2 original_champion   7.038885  3.278519    0.645833            0.498029        0.231968
2026-03-14     2    revin_champion   9.913551  3.166194    0.822917            0.701422        0.224021
2026-04-26     3 original_champion 110.198695 10.016197    0.968750            0.787087        0.071540
2026-04-26     3    revin_champion 115.351331 10.662571    0.989583            0.823890        0.076157

Source protocol/frozen_components.json and audit.json document input-only scales, learned-affine residual reversal, causality, exact weights, checkpoint replay, ordered quantiles, hourly conservation and CQR validation. Full predictions/checkpoints retained remotely in the recovery archive.
