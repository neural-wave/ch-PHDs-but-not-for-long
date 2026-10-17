# Two segments, eight daily coefficients — 2026-10-17

User authorized testing four predicted Fourier coefficients for the first
segment and four for the rest, with a tuned annual sinusoidal boundary.

Compare matched eight-output predictors: global Fourier8, fixed-noon 4+4,
learned-constant-boundary 4+4, and learned-seasonal-boundary 4+4. Training-only
greedy conditional SSE at noon selects four distinct fixed-phase harmonics per
segment from 1..12 cycles/segment, sine or cosine. Freeze these modes across
segmented alternatives. The global control repeats the previous train-selected
four-frequency free-phase/eight-output basis.

Each segment's Fourier functions use time relative to its own start and length.
A sigmoid first-segment gate, scale0.5h, weights them, and its complement weights
the second segment. This is a smooth overlapping transition, not a hard cut.
Project every resulting basis function to zero within each hour and normalize
its L2 length; final residual projection also enforces hourly conservation.

The seasonal boundary is b0+bs*sin(2*pi*doy/365.25)+bc*cos(...), shared across
zones. A constrained parameterization keeps b0 in9..15h and the full annual
boundary in8..16h. Constant variants have one shared trainable boundary
parameter; seasonal has three, beyond the eight daily coefficients. Daily
coefficients use causal rich history/known current hourly input features,
flattened across the day, with train-only normalization and zone identity.
No quarter-hour target of the day is a predictive input.

Fit302 days October–July. Select epochs/configs with August31; September30 is
development evaluation, already viewed. Boundaries are fitted jointly using
training reconstruction MAE, with no September optimization. This first test
does not perform blocked seasonal cross-validation; one annual cycle and
August-only stopping limit conclusions about seasonal generalization.

Search linear and64-hidden SiLU MLP predictors, learning rates .001/.003,
seeds17/43. Constant/seasonal boundary initializations10/12/14h; fixed/global
one initialization. Total64 fits. AdamW decay.01 for predictor; boundary
learning rate.02, no decay;300epoch maximum,30patience,32-day minibatches.
Selection saved before September predictive scores and oracle capacity.

Exact L1 oracle weights on selected boundaries diagnose capacity only, never
enter training features or model selection. Smoke exercises all modes, checks
nonzero boundary gradients and finite-difference gradients for the seasonal
basis. Serialized selected-model coefficient replay is exact. Independent
NumPy artifact audit reconstructs seasonal boundaries/bases, verifies eight
daily coefficients, hourly means, all reported MAEs and oracle lower bounds.

Sources: segmented_frequency.py, run_segmented_frequency.py,
audit_segmented_frequency.py. Output results/segmented_frequency_20261017,
including selected checkpoints, daily coefficients/predictions/bases/boundaries,
all-trial metadata, manifest, selection, metrics, report, audit and figure.
Execute on the existing authorized root@93.91.156.85:55340 RTX5090 instance,
/workspace/HACKATON, via current remote Nightwatch. No deployment/refit.
