# Minimal joint four-zone CNN — 17 October 2026

Supersedes the broad architecture search as the current requested experiment.
Fixed architecture: Conv1d(input_channels,16,kernel_size=5), ReLU,
Conv1d(16,4,kernel_size=1). One hidden layer, no dilation, attention, pooling,
dropout, residual blocks, ensemble, or architecture search. Input channels have
fixed zone identities; each convolution mixes all four zones simultaneously.

Current-day hourly prices are repeated to 96 quarters. Four one-hot quarter
position channels distinguish the position inside each hour. The matched history
condition adds exactly seven previous calendar days of raw quarter prices,
repeated hourly means and missing-day masks. There are 8 input channels without
history, 71 with it; exact trainable parameter counts are 724 and 5,764.
Interpolation is the fixed output anchor. Predicted within-hour residuals are
mean-centered, preserving hourly means. No extra calendar or engineered feature
channels. History is strictly prior; missing dates are not compressed.

Six fits only: three seeds (17/43/71) for each history condition, width16,
depth1, kernel5, learning rate .001. Seeds are selected using August only,
without averaging models. Shared training defaults remain AdamW, MAE, batch32,
maximum300 epochs, patience30. Train302/August31/September30; September is
developmental, not blind. Train-only normalization; no deployment refit.

Run: code/run_simple_joint_zone_cnn.py sets JOINT_CNN_SIMPLE=1 for shared code and
audit. Outputs results/simple_joint_zone_cnn_20261017. Source code/joint_zone_cnn.py
retains the old architecture as its default for checkpoint compatibility.
Both training and inference disable TF32. The earlier broad search's final audit
failed because inference had inconsistent TF32 settings; replaying its saved
checkpoints with matching settings passed the independent audit without retraining
or changing August selection. Broad-search September MAEs:5.862788 without
history,5.485715 with history. These are separate from the minimal model.
