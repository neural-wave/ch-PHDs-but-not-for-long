# HACKATON project memory

Objective (2026-10-16): minimize quarter-hour reconstruction MAE using official inputs only; no pretrained/external/October prices. Four zones jointly; hourly means preserved within0.0001 EUR/MWh. Monthly split: Oct2025–Jul2026 train302 days, August31 validation, September30 development test (previously viewed). Official data363 days/139,392 quarters.

Verified prior monthly winner monthly_joint_blend_gpu1: August4.245329, September5.091209; September serialized/CSV audit passed11,520 unique finite rows and strict prior history. Evidence results/monthly_joint_blend_gpu1/audit.json, results/monthly_best.json; report results/experiments_summary.html. RevIN wins2/4 paired comparisons, no broad superiority. Original startup summary preserved in memory/records/20261016_before_hierarchical_startup.md.

Separate frozen all363 deployment refit on Ando GPU1 was running at last check, not rechecked this conversation: Nightwatch816b1393f18c891ba8a8de11, supervisor2183339/process2184735; remote /home/mnogales/Projects/HACKATON/results/deployment_all363_gpu1. See DEPLOYMENT_REFIT.md. No held-out deployment score; monthly inference manifest preserved.

New hierarchy/conformal experiment on Vast root@ssh1.vast.ai:24199, /workspace/HACKATON; RTX PRO4000 Blackwell. Eight neural hierarchy/direct candidates plus interpolation. Same302-day fitting; August1–15 selection, August16–31 conformal calibration; September evaluation. See HIERARCHICAL_PROTOCOL.md. Nominal intervals require empirical coverage checks under dependence/drift.

GPU preflight and complete two-candidate smoke/replay/CSV audits passed; full campaign started. Initial fixture failure57114bfaba49bf1217220b0b corrected. Restart5c9227f73977427b05cb824b running with heartbeats/Telegram, worker1380; log /root/.local/share/nightwatch-hackaton/5c9227f73977427b05cb824b.log. ETA23:22 Zurich, October16. Results results/hierarchical_vast; no full results yet. Next verify terminal state, retrieve/audit MAE and coverage/widths; never infer success from ETA.
