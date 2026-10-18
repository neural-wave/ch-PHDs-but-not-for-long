# Four daily residual weights — 2026-10-17

User authorized testing four daily frequency weights. Reconstruct each zone's
day as mean-corrected interpolation plus a four-coefficient residual basis.
Subtract every basis function's within-hour mean, preserving the given hourly
prices. Four means four real outputs per day **per zone**, not four trainable
parameters for the whole predictor.

Training: 302 October–July days; August: 31 selection days; September: 30
development evaluation days, previously viewed. No external or October target
data. Existing rich inputs are recomputed causally, flattened across the known
day, scaled using training observations and appended with zone identity.
Perturbing current/future targets must leave first-September inputs unchanged.

Training-only greedy squared-error selection chooses four distinct Fourier
frequencies, each with fixed sine/cosine phase. Each projected candidate is
orthogonalized against selected candidates for conditional selection. Report
the actual frequencies and phases; orthonormalized coefficients span these
four original projected functions. This is a fixed dictionary, not prediction
of new frequency locations each day. A free-phase control includes both sine
and cosine at selected frequencies (at most eight independent outputs).
Uncentered training-residual SVD supplies rank 4/8/16 controls; rank 72 checks
the full mean-zero residual space. Shared bases are fitted across zones.

Exact linear-programming L1 oracle weights on held-out targets diagnose
approximation capacity; they are never predictive inputs. Train linear and
64-unit one-hidden-layer MLP coefficient predictors directly on reconstruction
MAE: learning rates .001/.003, seeds 17/43, at most 300 epochs, patience 30,
AdamW decay .01. Select candidates/epochs by August only, save selection before
predictive September scoring. No ensemble or refit.

Artifacts: results/four_frequency_20261017/{manifest.json,basis.npz,
selection.json,metrics.json,REPORT.md,independent_audit.json,comparison.png}.
Save weights, per-day coefficients and predictions for replay. Independent
audit recomputes MAE, hourly conservation, oracle bounds, basis-span membership
and the four-output constraint. Source: code/frequency_experiment.py,
code/run_frequency_experiment.py, code/audit_frequency_experiment.py. Execute on the
existing authorized RTX 5090 instance under remote Nightwatch; completion means
the numerical audit passed, with local visual inspection separate.
