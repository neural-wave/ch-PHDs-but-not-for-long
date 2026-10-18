---
title: Minimal joint CNN completed; 2D CNN layout clarification pending
status: current
kind: result-and-proposal
recorded_at: '2026-10-17T10:30:38.869116+00:00'
sources:
- results/simple_joint_zone_cnn_20261017/REPORT.md
- results/simple_joint_zone_cnn_20261017/audit.json
- SIMPLE_JOINT_ZONE_CNN_PROTOCOL.md
experiment_id: simple_joint_zone_cnn_20261017
supersedes: []
---

2026-10-17 verified remote minimal six-fit campaign completed with independent audit passed, both training-only normalization and checkpoint replay verified. August-selected seed17 for both history conditions. No-history724params: August4.90670903/September5.92182411; seven-day history5764params: August4.71624431/September5.53999604. Hourly maximum error5.684e-14. Entire minimal experiment directory copied from root@93.91.156.85:55340:/workspace/HACKATON/results/simple_joint_zone_cnn_20261017 to local results/ including six checkpoints and selected predictions; no deployment. Nightwatch799366803bf6fbf860494899 log terminal AUDIT PASSED. User now requests simple 2D CNN with kernel8 on time/representation axis, several days potentially week, all zones and extra last-layer inputs; asks U-Net vs classification. Clarification requested: concatenate96+24+96 columns vs aligned96 streams and separate24branch; eight day rows (7past+current masked), and final covariates current hourly, neighboring-hour changes, quarter position, weekday. No 2D training launched until input layout clarified. Recommendation is continuous regression; U-Net is architecture distinct from loss/task. Kernel plots alone are not complete interpretation; plan heatmaps plus activations and ablations.
