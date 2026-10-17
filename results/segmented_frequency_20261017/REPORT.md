# Two-segment Fourier reconstruction — 2026-10-17

Eight predicted coefficients per day/zone. Training through July; August selection; September developmental.
Shared seasonal boundary fitted jointly on training MAE; three seasonal parameters beyond daily coefficients.
Smooth overlap scale 0.5h; segment-relative time stretches basis functions; exact hourly-mean projection.

| Model | August MAE | September MAE | September oracle MAE | b0 (h) | sin (h) | cos (h) |
|---|---:|---:|---:|---:|---:|---:|
| global8 | 4.923514 | 5.947649 | 4.903183 | 12.000 | 0.000 | 0.000 |
| noon | 5.545422 | 6.397598 | 5.625719 | 12.000 | 0.000 | 0.000 |
| constant | 5.304453 | 6.134637 | 5.278974 | 9.257 | 0.000 | 0.000 |
| seasonal | 5.335967 | 6.124439 | 5.287937 | 9.368 | 0.051 | -0.275 |
| interpolation | 6.238029 | 7.090321 | nan | nan | 0.000 | 0.000 |

Modes per segment: [[[1, "sin"], [12, "sin"], [2, "sin"], [3, "cos"]], [[1, "sin"], [11, "cos"], [2, "cos"], [3, "sin"]]]
Oracle weights use true targets solely to diagnose capacity. No deployment or refit.
One annual cycle and August-only stopping limit seasonal evidence. Sine shape is a hypothesis, not an established physical boundary.
