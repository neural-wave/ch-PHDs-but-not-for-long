# Paired RevIN ablation (2026-10-16)

Eight candidates: widths/blocks 64/1 and 64/2, seeds 17 and 43, with and without reversible normalization; learning rate 0.0003. Same 302 training days, August checkpoint selection, maximum 250 epochs with early stopping. September remains a development test.

Each model receives all four zones jointly through temporal convolution and node attention. Both receive the existing 389 training-standardized rich features plus seven explicit hourly channels: the zone's current hourly value, all four concurrent zone values, daily mean, and daily standard deviation. These additional channels have identical dimensions in both variants.

RevIN uses the mean and population variance of each zone's supplied 24 hourly inputs only; scale is sqrt(variance + 1 EUR²/MWh²), guarding flat inputs. A learned per-zone affine transformation precedes the network. The head predicts zero-mean quarter deviations in normalized units; undo affine scale and multiply by the input-derived scale, then add the mean-compliant interpolation baseline. The affine offset cancels under the final zero-mean projection. The control uses hourly prices divided by 100 and the existing fixed output residual scale 20. Absolute mean/scale side features and original rich features retain cross-zone level information, making this a hybrid RevIN ablation rather than a claim that all engineered inputs are instance-normalized.

Optimize original EUR/MWh MAE in both cases. cuDNN and matrix TF32 are disabled in this new run. No target-derived normalization statistics or online fitting. Mix candidates with the current parent ensemble using August only; retain the parent contender. Full results must pass saved-model inference checks before replacing the frozen manifest.

Implementation: code/revin_models.py, code/train_revin.py, code/predict.py; remote source snapshot revin_campaign/ under /home/mnogales/Projects/HACKATON. Tests verify isolated day statistics, finite gradients, flat/negative prices and zero residual means. Two-epoch smoke exported ensemble, control-only and RevIN-only CSVs successfully.

Full run Nightwatch: 845d62f24e76c25572da6009, Ando physical GPU1, supervisor 2175934, ETA 45 minutes from 2026-10-16 20:28 UTC. Worker log /home/mnogales/.local/share/nightwatch-worker-bitsvsdims/845d62f24e76c25572da6009.log. First launch afa69e6919db86ea0674afc3 exited 2 before training because its upload failed; the upload was repaired and its terminal state verified before relaunch.

Reference: https://github.com/ts-kim/RevIN (ICLR 2022 official implementation). This adaptation is not an identical replication of the paper architecture.

## Verified results

Run 845d62f24e76c25572da6009 completed exit 0. RevIN beats its matched control in two of four settings; average standalone August difference -0.006642 EUR/MWh, driven by the two-block seed43 model. That component receives about 3.4%, 10.7%, 10.7%, 0% zone weights in the parent mixture. August MAE 4.246177, September 5.091086. A descriptive day-bootstrap interval for September improvement is [-0.00702,+0.00421] EUR/MWh, spanning zero; it is not a valid post-selection significance test. See results/monthly_revin_gpu1/revin_analysis.json. Full-month serialized replay passed.

A follow-up jointly reweighted the existing selected leaves and retained the frozen parent fallback. It selected on August only, giving August 4.245329 and September 5.091209. This wins August despite its slightly worse September score. Its whole-month replay passed, with the same documented original-graph TF32 cache drift. Nightwatch dfc5c8021bec7248371f2f87 completed exit0. See code/train_joint_blend.py and results/monthly_joint_blend_gpu1/audit.json.
