---
title: Confidence and catastrophic error detection set aside by user 20261017
status: superseded
kind: decision
recorded_at: '2026-10-17T13:16:29.827821+00:00'
sources:
- results/quantile_2d_cnn_20261017/validation_risk_diagnostic.json
- results/quantile_2d_cnn_20261017/confidence_metrics.json
- results/quantile_2d_cnn_20261017/REPORT.md
experiment_id: quantile_2d_cnn_20261017
supersedes: []
status_updated_at: '2026-10-17T14:07:44.946398+00:00'
---

On17October2026, after reviewing the lack of catastrophic examples and the missed September5 warning, the user said: "okey then it seems it does not work, lets leave it aside". In the immediate conversation this refers to the confidence / catastrophic-error detection branch. Set that branch aside: no further fitting, calibration, model search or deployment of confidence/risk probes unless the user reopens it. This is a user-directed stop on that branch, not evidence that all possible error detection methods fail.

Preserve the audited quantile reconstruction result: analog90 quantile-cohort model trial23, SeptemberMAE5.216650727743593, exact hourly conservation. The user had accepted approximately5.2 as good enough. No instruction to discard that CNN, delete artifacts or alter deployment. Keep all risk analysis artifacts and caveats as historical evidence, including the failed warning for September5 (actualMAE7.243371, predicted4.807541, rank28/30), the1/182earlier and0/31August/0/30September severe events, and the distinction between interval coverage and error-warning reliability. No active job exists for this completed campaign.
