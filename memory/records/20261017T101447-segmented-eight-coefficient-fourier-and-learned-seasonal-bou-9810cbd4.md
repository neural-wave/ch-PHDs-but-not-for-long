---
title: Segmented eight-coefficient Fourier and learned seasonal boundary launched
status: current
kind: experiment
recorded_at: '2026-10-17T10:14:47.977020+00:00'
sources:
- SEGMENTED_FREQUENCY_PROTOCOL.md
- segmented_frequency.py
- audit_segmented_frequency.py
- run_segmented_frequency.py
experiment_id: 270e46a91311c053123bd97a
supersedes: []
---

2026-10-17 user authorized try of two four-coefficient segments with annual sine-wave boundary. Implemented and launched on live-verified idle existing RTX5090 root@93.91.156.85:55340, /workspace/HACKATON, /venv/main/bin/python (torch2.14.1+cu130, scipy1.18.1). Nightwatch health passed and current client copied before launch. Nightwatch270e46a91311c053123bd97a, worker4101, /root/.local/share/nightwatch-hackaton/270e46a91311c053123bd97a.log; initial running heartbeat_expected:true, timeout180s, ETA10min. Wrapper runs smoke, full64 fits, independent audit; actual terminal success/failure alerts enabled. No completion claim at record creation.

Source hash and causal features checked in job. Training302 throughJuly, August31 selection/stopping, September30 developmental evaluation only; no September fitting. All modes eight coefficients/day/zone from same rich causal history+known current hourly inputs. Compare global8, smooth-noon4+4, learnedconstant4+4, learnedseasonal4+4. Train-only greedy selection at noon selects four distinct fixed-phase harmonics from1..12 per segment; modes frozen for segmented alternatives. Segment-relative Fourier time stretches with its duration, sigmoid transition scale0.5h, per-basis hourly projection and L2 normalization. First and second gates sum to1 but smoothly overlap; not strictly disjoint segments.

Constant boundary has one shared parameter, seasonal b0+bs*sin(annualday)+bc*cos(annualday) has three shared parameters. Constrained b0 within9..15 and seasonal full curve within8..16h. Boundaries jointly fitted on training MAE. Two seeds17/43, learning rates.001/.003, linear and64hidden MLP, initialboundary10/12/14 for learned variants; global/noon single initial12. Maximum300epochs,patience30,AdamW weightdecay.01 predictor; boundarylr.02,no decay. This first test uses August validation, not blocked seasonal CV; annual generalization remains uncertain with one yearly cycle.

Smoke covers all four modes, nonzero boundary training gradients, seasonal finite-difference gradient, coefficient checkpoint replay. Full outputs results/segmented_frequency_20261017/{manifest.json,normalization.npz,selection.json,trials.json,metrics.json,REPORT.md,independent_audit.json,comparison.png}; selected model.pt and outputs.npz per mode contain coefficients,bases,boundaries and predictions; exact L1 oracle diagnostic target-derived weights computed only after predictive selection frozen. Independent NumPy replay checks sinusoidal boundaries and basis, eight-output shapes, conservation and MAEs. No deployment/refit. Next verify smoke/full state, retrieve report/compact artifacts and visually inspect.
