# HACKATON project memory

Objective: official-only four-zone quarter-hour reconstruction; preserve hourly means within0.0001 EUR/MWh. No external/pretrained/October prices; strictly earlier history. Train302 days throughJuly, August31 selection, September30 developmental evaluation, not blind. Zones anonymized.

Verified2026-10-17: monthly_joint_blend_gpu1 SeptemberMAE5.091209 versus interpolation7.090321; offline363-day refit audited, no external submission. DEPLOYMENT_REFIT.md; results/COMPLETION_AUDIT.md. Simple linear/additive/MLP/CNN5.209794/5.255107/5.343947/5.222494; results/simple_5090. Hierarchy5.282777; lagHPO retained parent.

Active2026-10-17: user confirmed joint CNN, all4 zones simultaneously, matched history-off versus previous7 calendar days of raw96-quarter and24-hour curves. 144fits; joint96-step dilated convolution, interpolation anchor, hourly conservation. JOINT_ZONE_CNN_PROTOCOL.md; results/joint_zone_cnn_20261017. VastRTX5090 root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python. Nightwatch54daabf3443590d977ed8fb5 worker4472, heartbeat180s, initialETA45min; log/root/.local/share/nightwatch-hackaton/54daabf3443590d977ed8fb5.log. Smoke passed: lag values, masks, causal perturbation, cross-zone gradients, conservation. Four workers beat eight in short benchmark; full search running, final audit pending. Next retrieve audited results.

Earlier segmented64fit run270e46a91311c053123bd97a results pending/unrechecked; SEGMENTED_FREQUENCY_PROTOCOL.md. Four-frequency study audited, no deployment: FOUR_FREQUENCY_PROTOCOL.md; results/four_frequency_20261017. Earlier exact results/pointers preserved in memory/records/20261017_startup_before_joint_zone_cnn.md.

Other context: analog residual history dominates reliance; ordered7-day interpolation-error ablation unlaunched. INTERPOLATION_HISTORY_PROTOCOL.md. Quantile16fold median5.226105; results/linear_quantiles_vast. Expanding/conformal034d5133ebc5e076af6aed88 final audit outstanding. User prioritizes interpretability, analogs, difficult-day robustness; INTERPRETABLE_ROBUSTNESS_PLAN.md. SSM untested.
