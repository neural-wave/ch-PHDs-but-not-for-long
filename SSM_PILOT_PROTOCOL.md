> Historical exploratory pilot. The eligible comparison now follows VALIDATION_PROTOCOL.md; see campaigns/ssm_monthly_20261017/PROTOCOL.md and results/ssm_monthly_20261017/REPORT.md. These original 56 fits are not eight-fold results.

# S4D/S5 ordered-history pilot — 2026-10-17

Authorized by user: try state-space models on existing Vast GPUs. Run only on
existing root@93.91.156.85:55340, /workspace/HACKATON, RTX5090;
/venv/main/bin/python. No new instance or deployment change.

## Design

All four zones jointly, seven previous calendar days as168 hourly tokens.
Each zone supplies four signed and four absolute interpolation errors, availability
and hourly level. Three clock/lag channels complete43 history channels. Missing
calendar days retain24 masked tokens. Query has108 compact current/history-summary
channels over24 hours. Full current-day hourly inputs are known; no current-day
quarter targets enter features. Past evaluation targets become available only
on subsequent dates, following the existing jury-history protocol.

S4D diagonal SISO kernels per hidden channel, S5 diagonal MIMO dynamics with
HiPPO-normal eigenbasis initialization, GRU and causal dilated TCN share history
inputs, one residual encoder block, and the same two-convolution current-day
decoder. Last24 encoded history states plus pooled history condition the decoder.
Compact control keeps identical query features but omits sequential history.
Widths16/32; state32(real equivalent,16 complex modes); rates.001/.003;
seeds17/43.32 sequence fits plus4 compact fits. Report parameter counts: model
families are input/budget matched, not exactly parameter matched.

PyTorch task-specific S4D/S5 reimplementations, not official benchmark reproductions.
S4D uses S4D-Lin initialization, ZOH discretization and FFT convolution. S5 uses
HiPPO diagonal conjugate-half initialization, stable negative real eigenvalues,
ZOH and autograd-safe affine parallel scan. SSM timescale/dynamics/input parameters
use0.1× learning rate and zero decay. Other parameters AdamW decay.01.
Three zero-sum contrasts produce interpolation corrections; optimize quarter MAE.
Float64 final reprojection. No quantile/confidence claim.

## Selection and verification

302 fitting days; August31 days select hyperparameters and epochs; September30
days development only. Maximum160 epochs, patience20, batch32, clip5.
Four earlier fixed-config folds evaluate index ranges[181,212),[212,243),
[243,273),[273,302), with preceding15 days for epoch selection and earlier days
for fitting. Fixed width32/state32/rate.001/seed17 for all five families; no reuse
of August-selected hyperparameters.20 additional fits;56 total.
Every fold uses its own fit-only normalization. Data hash, calendar source audit,
current/future target perturbation, recurrence equivalence, finite gradients,
stability, saved-checkpoint replay and hourly conservation are required.
Report official MAE, zone errors, fixed common difficult-day subset, parameters,
wall times, and earlier chronological fold errors. All historical months remain
development evidence. No September-driven search or automatic deployment.

Run via current remote Nightwatch wrapper. Expected artifacts:
results/ssm_pilot_20261017/{manifest,smoke,progress,selection,rolling,metrics,audit}.json,
REPORT.md, trials/*/{model.pt,curves.csv,predictions.npy,metrics.json}, rolling/*/.
Completion requires audit.json passed. Retrieve/back up results after completion.

References: https://arxiv.org/abs/2206.11893 ;
https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py (Apache-2.0);
https://arxiv.org/abs/2208.04933 ;
https://github.com/lindermanlab/S5 (MIT). Original S4 not included in this bounded
pilot; S4D is explicitly identified as its diagonal variant.
