# Protocol-compliant S4D/S5 quantile comparison — 17 October 2026

Supersedes the September-focused pilot as the eligible comparison. Its completed
56 fits in results/ssm_pilot_20261017 remain exploratory and are not eight-fold
results. No checkpoint/preprocessing is reused from that pilot.

Follow VALIDATION_PROTOCOL.md exactly: eight February–September monthly tests;
initial fitting October–December2025; previous-month days1–15 select,16–end
calibrate; expand training monthly, never refit on selection/calibration dates.
Train preprocessing/model from scratch per fold; test parameters/calibration stay
frozen. Only strictly earlier observed quarters enter history. Missing DST dates
remain omitted. Full known current hourly curves permitted; official data only.

Five families: S4D, S5, GRU, dilated causal TCN, compact-summary CNN control.
Same168hour prior-seven-calendar-day representation, joint four-zone inputs,
compact query features and common decoder as SSM_PILOT_PROTOCOL.md. Compact
control omits the sequential history encoder. Task-specific PyTorch S4D/S5
implementations; S4D is the diagonal variant, not full original S4. Parameter
counts differ and are reported. Fixed width32/state32; rates.001/.003;
seeds17/43.20 candidates per fold,160 full fits. Maximum300epochs/patience35;
AdamW and zero-decay,0.1×rate SSM dynamics groups; batch32,clip5,dropout.1.
Ordinary training: conserved-median MAE + mean pinball across all eleven levels.
No second month-weighted training objective in this bounded comparison.

Every candidate outputs ordered .025/.05/.10/.20/.25/.50/.75/.80/.90/.95/.975
quantiles. q50 uses three zero-sum contrasts and is the coherent point decision.
Positive increments enforce ordering around it; fitting-only empirical residual
gaps initialize the uncertainty head. All candidates select epochs/configuration
by first15day median MAE/interpolation MAE. Save selections before test scoring.
No ensembles, outer-test-driven tuning, or deployment replacement.

Use existing campaigns/monthly_quantiles_20261017/trainq.py evaluation helpers:
nonnegative CQR widening by zone/quarter with finite-sample ranks,16–end calibration
only, nested50/80/90/95%intervals. Save raw quantiles, all calibrated endpoints,
width percentiles and keyed CSV for validation selector. Confidence bins describe
relative calibration-width, not correctness probabilities. Report pinball,
coverage, width and width/error associations; no catastrophe classifier.

Primary score arithmetic mean of eight monthly median MAE / mean-corrected
linear-interpolation MAE ratios. Also each monthly ratio, worst ratio, pooled MAE,
distinct pooled MAE ratio, per-zone MAE/ratios, and alternate explicitly named
uncorrected-interpolation ratio. February–September241days/92544quarters.
All are developmental backtests, not blind independent folds.

Gates: first/last-fold all-family two-epoch GPU smoke; independent chronology,
train-only stats, source hashes, all candidate checkpoint/selection replay,
ordered quantiles, coherent median, independently recomputed calibration ranks,
coverage/width, primary and pooled score, and keyed exports. Complete only after
full audit and recovery archive. Run existing VastRTX5090
root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python, current detached
Nightwatch wrapper; reporting on real completion/failure. Preserve prior outputs.
