---
title: S4/S4D and S5 feasibility for constrained price reconstruction
status: current
kind: research
recorded_at: '2026-10-17T09:31:03.218842+00:00'
sources:
- simple_search.py
- INTERPOLATION_HISTORY_PROTOCOL.md
- results/feature_importance_20261017/ANALYSIS.md
- results/challenge_rules_extracted.txt
- INTERPRETABLE_ROBUSTNESS_PLAN.md
experiment_id: null
supersedes: []
---

Exploration requested 2026-10-17. Research/design only; no SSM implemented, training launched, resources rented or improvement measured.

Recommendation: prioritize a small S4D history encoder, then an S5 comparator. Full original S4 is a secondary complexity-control experiment. These are train-from-scratch models using official data only. They address ordered history representation rather than forecasting unknown hourly prices.

Verified project evidence: official rules lines85–152 give all24 target-day hourly prices together, permit prior provided quarter-hour history, require hourly conservation, and describe363 valid days across four anonymized zones. simple_search.py uses302 train/31 August selection/30 September development days, train-only normalization, MAE loss, serialized replay and three zero-sum contrasts. INTERPOLATION_HISTORY_PROTOCOL.md supplies seven calendar-day ordered errors as63 channels; CNN runs over24 hours, not over the672 quarter-hour history steps. Feature-importance report covers frozen simple models only: analog residual history mattered more than exact lag2/7; permutation evidence is not a retraining result.

Primary literature consulted:
- S4: https://arxiv.org/abs/2111.00396 ; official implementation https://github.com/state-spaces/s4 . Structured SSMs support convolutional sequence processing. Original fast kernels involve custom CUDA/PyKeOps integration; preserve special optimizer groups for state parameters.
- S4D: https://arxiv.org/abs/2206.11893 ; official minimal implementation https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py . Diagonal dynamics with careful initialization reduce implementation burden; published competitive results are not evidence of superiority on this dataset.
- S5: https://arxiv.org/abs/2208.04933 ; https://github.com/lindermanlab/S5 . Single multi-input/multi-output SSM versus S4's independent single-input/single-output systems, with parallel scans. Official code uses JAX; current project uses PyTorch. MIMO is a plausible fit to cross-zone channels but does not uniquely enable cross-zone mixing: S4D with learned channel mixing can also do it.

Suggested architectures:
A. Controlled24-hour replacement: same compact7 inputs as CNN, replace temporal convolutions with1–2 small S4D/S5 blocks, retain interpolation anchor and zero-sum head. Bidirectional processing of known current-day covariates is allowed. This isolates architecture but gives little long-sequence advantage.
B. Preferred hypothesis: encode prior7 days as672 chronological quarter tokens (or168 hour tokens with quarter-error channels as a cheaper first version). Inputs: signed/absolute interpolation residuals, known historical hourly levels/ramps, time/calendar, zone or cross-zone channels, availability and elapsed calendar time. Fuse history representation with a bidirectional24-hour current-curve decoder. A pooled history vector is a first baseline; per-hour aligned history readout is a separate ablation. Do not silently equate hour-token and quarter-token models.
C. Only if B helps: extend7→14→28 days or add current-curve conditioning of history readout. State-space dynamics alone are time-invariant and may not retrieve similar past curves as effectively as explicit analog matching.

Conservation: let interpolation baseline b have mean H. Head outputs (a,u,v), maps to correction (a+u,a-u,-a+v,-a-v), scaled consistently with existing /20 normalization. Train on reconstructed quarter MAE. Reproject in float64 at inference and assert tolerance. Loss must not be coefficient-wise MAE.
Information boundary: historical residuals require dates strictly before target day; unknown target-day quarters are never input. If appending current tokens, mark residual channels unavailable rather than filling with real targets. Both scan directions are permissible within the complete known covariate window; neither may cross into future target labels. Preserve DST calendar gaps using existing masks and gap/time encoding; do not compress missing days. Replay and perturb current/future targets with fixed hourly inputs to verify invariance.

Suggested staged study:
1. Unit/numerical checks: discretization/stability, scan-vs-sequential recurrence for S5, convolution-vs-recurrence where implemented for S4D, gradients finite, zero-sum output, source-date audit and serialized replay.
2. Fixed split pilot: S4D and S5 width32/state32 and width32/state64,1–2 blocks, modest dropout and separate state optimizer groups; small predeclared learning-rate choices and matched seeds. Start168-hour representation before672-quarter variant to bound engineering/runtime.
3. Controls: interpolation; compact7 linear/CNN; parameter-comparable GRU or temporal CNN using identical sequence inputs; simple exponentially decayed history; existing explicit analog model. Same features, normalization boundaries, early stopping budget and scoring. CNN compact7 alone is not a matched temporal-history control.
4. Select pilot architecture/hyperparameters on August; September remains already-inspected development evidence. Broaden surviving frozen candidates to chronological expanding folds, with tuning nested in each fitting period or clearly label reuse as development. Avoid random day splits/overlapping-window label leakage. History during evaluation may include newly observed past targets only when matching the jury's availability policy.
5. Report official MAE, per-zone/month, common fixed difficult-day subsets, paired daily differences/block uncertainty, parameters, measured runtime and offline dependencies. Standalone SSM versus conservative residual blend should use chronological out-of-fold blend selection. No speedup or gain estimates before measurement.

Interpretability: the bare linear SSM permits impulse kernels, mode decay times and state analysis; nonlinear stacked S4/S5 are not intrinsically interpretable prediction models. Maintain the explicit analog/additive direction and use SSMs as challengers unless reproducible gains justify added complexity. No confidence/calibration claims follow from hidden states.
Next step: implement a bounded matched-input pilot on an explicitly authorized remote resource; any long training requires Nightwatch registration and reporting. Existing campaign state was not inspected or changed during this exploration.
