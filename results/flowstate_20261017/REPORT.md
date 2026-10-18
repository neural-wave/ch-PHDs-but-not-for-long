# Exploratory FlowState reconstruction — 17 October 2026

External pretrained weights; excluded from the challenge submission. September is developmental, not blind.

| Method | September MAE (EUR/MWh) |
|---|---:|
| conditional_control | 5.454209 |
| conditional_flowstate | 5.461774 |
| flowstate_projected_only | 7.109678 |
| interpolation | 7.090321 |
| hourly_mean | 10.695366 |

Both conditional CNNs receive the entire current-day hourly curve, its slopes, quarter/weekday flags, and seven prior calendar days. The matched control has identical capacity with FlowState channels zeroed. FlowState r1.1 is frozen; 28 calendar days feed its native median 96-quarter forecast at scale0.25. The conditional CNN predicts mean-centered corrections around mean-corrected interpolation.

Fit: October2025–July2026 after28day warmup. August selects stopping and one seed per arm. Three seeds17/43/71 per arm; no September choice. Audit passed.

Evidence: metrics.json, audit.json, selection.json, model_receipt.json, predictions.npz and trials/*/model.pt.

Pretraining-data overlap with official historical prices has not been audited. This comparison measures these particular adaptations; it does not establish FlowState’s best attainable error.
