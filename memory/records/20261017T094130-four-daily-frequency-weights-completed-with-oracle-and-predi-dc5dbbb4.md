---
title: Four daily frequency weights completed with oracle and predictive audit
status: current
kind: experiment
recorded_at: '2026-10-17T09:41:30.427068+00:00'
sources:
- FOUR_FREQUENCY_PROTOCOL.md
- frequency_experiment.py
- audit_frequency_experiment.py
- results/four_frequency_20261017/REPORT.md
experiment_id: dcae73b185fbc7a157711e13
supersedes: []
---

2026-10-17 user authorized experiment. Executed on existing authorized root@93.91.156.85:55340 RTX5090, /workspace/HACKATON, /venv/main/bin/python; installed SciPy1.18.1. Refreshed remote Nightwatch from current local source. Nightwatch dcae73b185fbc7a157711e13, worker3517, /root/.local/share/nightwatch-hackaton/dcae73b185fbc7a157711e13.log. Actual wrapper terminal finish exit0 and delivered:true verified; independent numerical audit passed. Approx60 seconds run, initial15-minute ETA conservative. No other jobs changed.

Regular363-day source hash verified. Train302 throughJuly; August31 selects weights/model/epochs; September30 diagnostic already viewed. Recomputed causal rich daily flattened features with train-only normalization and zone ID. Shared train-only residual bases: greedy distinct-frequency conditional SSE fixed-phase projected Fourier4 (2sin,1cos,4sin,3sin cycles/day), matching four free-phase frequencies/eight coefficients, uncentered residual SVD rank4/8/16; rank72 sanity. Two seeds, two learning rates, linear/64-hidden MLP;40 candidate fits. Four coefficients are per day per zone, not total predictor parameters. Selected MLP4 has598084 parameters; flattened rich input is large, so this is output compression rather than a minimal predictor. Zero within-hour basis means preserve supplied means.

September predictor MAE / exact L1 target-derived diagnostic oracle: Fourier4 6.210906 /5.788496; Fourier four freephase8 5.977431 /4.903183; PCA4 5.827434 /4.989501; PCA8 5.887884 /4.402627; PCA16 5.887545 /3.487367. Interpolation7.090321; existing monthly_joint_blend5.091209 is historical comparator, not rerun here. PCA72 oracle1.36e-13. August-selected predictor MAEs5.311575,4.908225,4.872709,4.831232,4.754150 respectively. Selection frozen before predictive September scoring. Oracle targets used only for representation capacity, never predictor input.

Audits: causal first-September perturbation, source hash, training-only basis/scaling, finite predictions, serialized exact coefficient replay, independent MAE recomputation and hourly conservation, basis-span membership, oracle<=predictor/interpolation, rank72 reconstruction. Artifacts remote/local results/four_frequency_20261017 include all trial checkpoints, predictions, coefficients, basis, selection, metrics, REPORT, independent_audit and comparison.png. Fixed-frequency rank4 improves interpolation but chosen representation cannot match current pipeline even with ideal coefficients; PCA4 capacity nearly matches current model yet learned coefficient prediction trails. No deployment or replacement. Basis choice is greedy SSE, not globally optimal MAE or adaptive daily frequency search; this does not rule out every four-frequency model. Next possible direction: input-conditioned/local residual basis or better causal coefficient predictor, rather than assuming more outputs alone improve forecast.
