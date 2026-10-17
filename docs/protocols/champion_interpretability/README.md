# Possible interpretability studies for the champion ensemble

Prepared 17 October 2026. **Proposals only; no new study has run.** Target: `minimize_expanding_20261017/fixed_blend_ratio`, original fixed-scale champion, without RevIN. Registry: [best_model.json](../../../best_model.json); [exact fold recipes](../../../results/model_ranking_20261017/champion_per_fold.json).

## Verified starting point

The ranking's 13 source-campaign summary, monthly-score and audit hashes match the saved receipts (17 October check). Champion ratio remains 0.707968, pooled MAE 4.103184 EUR/MWh. This checks saved-result freshness, not remote job state or a new prediction replay. Research results are updated; the all-history deployment refit remains pending.

Already completed: [champion interpretation](../../../results/interpretability_deep_20261017/REPORT.md), with eight-fold forecast replay, zone/curve/uncertainty slices, signed-error cancellation, correlations and fixed-checkpoint member removal. Exact feature contributions there concern a standalone September linear model, **not the champion's linear experts**. Existing member-removal intervals were not recalibrated. Preserve these scope distinctions.

## Proposed order

| Priority | Study | Main question | Evidence to produce |
|---|---|---|---|
| 1 | [Expert corrections](01_EXPERT_CORRECTIONS.md) | Which expert contributes which within-hour shape? | Exact additive decomposition, representative cases, monthly summaries |
| 2 | [Grouped input sensitivity](02_INPUT_SENSITIVITY.md) | Which available input groups change champion predictions? | Plausible perturbation response, baseline versus learned-correction split |
| 3 | [Historical analogs](03_ANALOG_PROVENANCE.md) | Which prior dates supply useful or misleading shape information? | Audited analog provenance, lag/distance diagnostics, retrieval interventions |
| 4 | [Uncertainty and disagreement](04_UNCERTAINTY.md) | How do disagreement, interval width and errors relate? | Calibration-fixed bins, conditional coverage, failure atlas |
| 5 | [Stability and distillation](05_STABILITY.md) | Are explanations stable, and can a simpler surrogate reproduce the blend? | Fold/seed stability and held-out surrogate fidelity |

Follow [shared protocol](PROTOCOL.md) for every study. Record planned/running/completed/blocked status and source hashes in each eventual result directory; a proposal is not a result. Start with priority 1 using existing outputs, then priority 2 on an authorized research server. No model promotion or deployment change is implied.
