---
title: Exploratory FlowState with current hourly conditioning completed audited at
  MAE5.461774 versus matched control5.454209
status: current
kind: experiment
recorded_at: '2026-10-17T13:19:14.056589+00:00'
sources:
- campaigns/flowstate_20261017/PROTOCOL.md
- results/flowstate_20261017/REPORT.md
- results/flowstate_20261017/audit.json
- results/flowstate_20261017/selection.json
- results/flowstate_20261017/model_receipt.json
- results/flowstate_20261017/launch.json
experiment_id: flowstate_20261017
supersedes:
- memory/records/20261017T131643-exploratory-flowstate-conditioned-on-today-s-full-hourly-cur-ae1a13f7.md
---

## Completed and verified 2026-10-17

User authorized an exploratory pretrained FlowState error comparison despite submission ineligibility, then explicitly required the current-day hourly curve to influence quarter-hour shapes. The resulting adaptation feeds the entire4x24hourly curve through a global Linear96->16 path, with local hourly prices/slopes and seven prior calendar days, into a9156parameter conditional CNN. FlowState-r1.1 remains frozen and provides8 additional forecast/shape channels. Matched control uses exactly the same architecture with these channels zeroed. Centered corrections around interpolation conserve all hourly means. See PROTOCOL.md and earlier launch record for exact pins/environment and source design.

Audited September2026 MAE(EUR/MWh), all30days/4zones/11520quarters:
- Conditional CNN control:5.454209205155623; zone errors4.3420628903/7.2927893729/5.4837657628/4.6982187947.
- Conditional FlowState+CNN:5.461773706267895; zone errors4.2941751919/7.3603364023/5.5629827662/4.6296004647.
- FlowState forecast projected to hourly means:7.109677770694097.
- Mean-corrected interpolation:7.090321234809028.
- Hourly-repeat baseline:10.695366319444444.
FlowState-minus-control:+0.0075645011, approximately+0.139%. This bounded adaptation produced no clear incremental advantage. Do not infer a universal negative result or an optimized FlowState error. September is developmental/not blind; pretrained-corpus overlap remains unaudited; no submission/deployment change.

August-only selected controlseed71/bestepoch106/validationMAE4.6412646770; FlowStateseed17/epoch88/validationMAE4.6297881007. Six fits total. Independent audit passed CSV hash, equal-zone scoring/baseline crosscheck, exact hourly means, train-only normalization, August-only selection persisted before September inference, causal input perturbation, checkpoint replay. Replaymaxerrors3.55e-15/control and7.11e-15/FlowState. Trained models' early-quarter shapes respond to distant current hourly changes; FlowState model also responds to its pretrained feature channels.

Nightwatch102ec862033e194179693214 completed with exit0 at13:17:11UTC(15:17:11Zurich), approximately52seconds after launch; initial15minute ETA was conservative. Verified completed via authenticated tracker API, heartbeat_expected true. Remote root@93.91.156.85:55340,/workspace/HACKATON, worker7144, log/root/.local/share/nightwatch-hackaton/102ec862033e194179693214.log. Full result archive, including every head checkpoint and daily frozen forecast, backed up locally at results/flowstate_20261017_completed.tar.gz; local/remote SHA256 bothcc72193301f8c9c303ff428d13237590c94bc746adc75b550b03576c333dec4d. Human report/audit/selection extracted locally at results/flowstate_20261017/. Full recursive output copy also requested. External foundation weights are recoverable by pinned HF SHA, not included in the result archive. No active job or further experiment is required for this request.

Prior quantile, analog, blend and submission work remains unchanged; verbatim prior startup summary is preserved in the superseded launch record.
