---
title: 2D tuning smoke and nonzero original-control parity passed; full training verified
status: superseded
kind: verification
recorded_at: '2026-10-17T11:32:18.498110+00:00'
sources:
- results/parallel_2d_tuning_20261017/smoke.json
- TUNED_PARALLEL_CNN_PROTOCOL.md
experiment_id: parallel_2d_tuning_20261017
supersedes: []
status_updated_at: '2026-10-17T12:05:10.421157+00:00'
---

2026-10-17 livecheck: allseven features prepared, smoke passed control_parity/causality/missing_masks/cross_zone_gradient. Strengthened originalcontrol parity by makingoutputhead nonzero beforecopying sameweights; exactforwardarrayequivalence independently passed onGPU. No trainingconfigs or sourcefeatures changed. Fulltraining confirmed9/42complete inlog atcheck. Measuredtrainable counts: control1388,width16=3796,clock3860,residual4884,analog7/90=4948,context15124. Mostvariants stayunder5kparameters; widerfusioncontext15k comparator explicit. Local smoke copied. Nightwatch038fce95c7be0dda153059b8 worker5203, root@93.91.156.85:55340,/workspace/HACKATON; expectedheartbeats180s, initial15minETA (13:46Zurich), finalresults/auditpending. NoSeptemberimprovementclaim.
