# Parallel quarter/hour 2D CNN — 17 October 2026

User clarified quarter-hour and hourly curves should be parallel for genuine 2D
kernels, and approved late inputs: current hourly price, neighboring-hour
changes, quarter position, weekday. Preserve the simple architecture preference.

Grid: four zone price channels plus one availability-mask channel. Each day has
two adjacent rows: observed96 quarter prices and24 hourly means repeated four
times to96. Native24 hourly values are represented exactly by constant repetition,
not concatenated into the time axis. With history, rows are day-7 Q/H through
day0 Q/H, shape(batch,5,16,96). Query-day Q row is always zero and unavailable,
even during training; query-day hourly row is known. Missing calendar days have
zero prices and masks, without changing lag positions. No-history grid has2rows.

Conv2d(5,8,kernel=(2,8),stride=(2,1)) spans quarter/hour rows together and a
two-hour window. Replicate padding3left/4right keeps96 time positions. ReLU gives
eight feature maps per day. Flatten day identities into channels (no averaging),
append23 late channels:4current hourly prices,4left changes,4right changes,
4quarter-position flags,7weekday flags. Conv1d(...,8,1),ReLU,Conv1d(8,4,1)
produces four continuous curves jointly. The small nonlinear fusion lets late
information interact with learned features. Exact parameters940 without history
and1388 with history. No dilation, pooling, U-Net, classification, ensemble or
architecture search. Day mixing happens in the small fusion layer; the 2D kernel
itself spans the two representations, not neighboring calendar days.

FixedLR.001, seeds17/43/71 for each history condition: six fits. Shared MAE/
AdamW defaults, train302/August31/September30. Normalization train-only; August
selects seed and epoch before September inference. September is developmental.
Interpolation anchor and within-hour residual centering preserve hourly means.
Official data only; no deployment refit. All paths disableTF32 consistently.

Sources: parallel_zone_cnn.py, joint_zone_cnn.py (JOINT_CNN_PARALLEL=1),
run_parallel_zone_cnn.py, audit_joint_zone_cnn.py. Run on existing VastRTX5090
root@93.91.156.85:55340,/workspace/HACKATON,/venv/main/bin/python, via Nightwatch.
Outputs results/parallel_2d_zone_cnn_20261017. Smoke checks exact paired rows,
hourly repetition, masks, prior-only histories, cross-zone gradient and output
means. Independent final audit reconstructs features/statistics, checks August
selection, metrics, means and checkpoint replay. Selected checkpoints export
learned_filters.npz including normalization and fusion weights, five kernel
heatmaps (four zones+mask), and activations for August1. Plots show weights for
normalized inputs, not causal explanations. Matched history ablation supplies
behavioral evidence; no quantitative kernel attribution claim.
