# Four daily frequency weights — 2026-10-17

Fit: October–July. Selection: August. September is developmental, not blind.
Oracle uses target-derived optimal weights and measures representation capacity only.

| Basis | Daily weights / zone | August predictor MAE | September predictor MAE | September oracle MAE |
|---|---:|---:|---:|---:|
| fourier4 | 4 | 5.311575 | 6.210906 | 5.788496 |
| fourier4_freephase | 8 | 4.908225 | 5.977431 | 4.903183 |
| pca4 | 4 | 4.872709 | 5.827434 | 4.989501 |
| pca8 | 8 | 4.831232 | 5.887884 | 4.402627 |
| pca16 | 16 | 4.754150 | 5.887545 | 3.487367 |
| interpolation | 0 | 6.238029 | 7.090321 | nan |

Fourier fixed-phase selections: [{"cycles_per_day": 2, "phase": "sin"}, {"cycles_per_day": 1, "phase": "cos"}, {"cycles_per_day": 4, "phase": "sin"}, {"cycles_per_day": 3, "phase": "sin"}]
Projected Fourier functions preserve hourly means; orthonormalization keeps their joint span.
The four outputs are daily coefficients; the predictor has additional trainable parameters.
Audits: source hash; causal first-September feature perturbation; train-only bases/scaling; finite outputs; hourly conservation; saved-weight replay; full-rank reconstruction; oracle dominates predictor.
