# Joint four-zone CNN — 17 October 2026

User confirmed one model producing all four zones together, with a matched
comparison of no history versus previous seven calendar days of both observed
quarter-hour prices (96/day/zone) and hourly means (24/day/zone).

Inputs have a 96-quarter temporal axis. Four current hourly curves are repeated
to quarter resolution, alongside their mean-corrected interpolation and calendar
and quarter-position channels. The history condition appends each lag's four
quarter-hour curves, four repeated hourly curves and a missing-day mask. Missing
calendar days are zero-filled, not skipped. Histories use only strictly earlier
observations, including rolling observed history during August/September. Hourly
history is derived from the same official observed quarter data. Current-day
quarter targets are never inputs. Current hourly means are official query inputs.

Conv1d mixes all zone channels with unrestricted weights; its four output
channels predict all four zones simultaneously. Dilated temporal layers use
widths 32/64/128, depths 2/4/6, kernels 3/5. Learning rates .001/.003, seeds
17/43, both history conditions give 144 fits. Interpolation anchors every model;
subtracting the four-quarter residual mean preserves every hourly mean exactly.
AdamW .01 weight decay, MAE, batch 32 days, gradient clipping 5, maximum 300
epochs and patience 30. Normalization uses only the first 302 training days.
August selects epochs/configuration separately for each history condition;
choices are saved before September inference. September has already been
inspected in development. No external/pretrained data or deployment refit.

Run on existing Vast RTX5090 root@93.91.156.85:55340, /workspace/HACKATON,
/venv/main/bin/python. Smoke validates exact lag values, future-target
perturbations, missing masks, cross-zone gradients and hourly conservation,
then profiles the same eight short fits at 4/8 concurrent workers. Full search
uses the faster measured setting. Nightwatch wraps preparation, smoke, training,
and independent final audit, with actual success/crash notifications.

Sources: joint_zone_cnn.py, run_joint_zone_cnn.py, audit_joint_zone_cnn.py.
Outputs: results/joint_zone_cnn_20261017/{manifest,smoke,selection,metrics,audit}.json,
REPORT.md, trials/*/{model.pt,metrics.json}, selected trials/predictions.npy.
