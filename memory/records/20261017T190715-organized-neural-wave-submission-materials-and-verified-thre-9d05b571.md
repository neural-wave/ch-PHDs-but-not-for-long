---
title: Organized Neural Wave submission materials and verified three-page draft
status: current
kind: submission_preparation
recorded_at: '2026-10-17T19:07:15.725579+00:00'
sources:
- delivery/README.md
- delivery/CHECKLIST.md
- delivery/RUN_FOR_JURY.md
- delivery/report/report.md
- delivery/report/report.pdf
- delivery/submission_status.json
- code/check_submission.py
- results/submission_organization_20261017/verification.json
- results/submission_organization_20261017/readiness.json
- submission/FREEZE_MANIFEST.json
experiment_id: null
supersedes: []
---

On 17 October 2026 the user supplied the general Neural Wave submission guidelines and asked to organize the submission. The user explicitly deferred full team names and roles until later. Added a delivery/ folder with a jury-facing README, final checklist, current frozen-bundle run guide/CSV contract, credits and AI-usage disclosure, machine-readable pending status, an editable report and visually inspected three-page PDF, a timed two-minute video script and a five-minute presentation outline. Added docs/reference/SUBMISSION_GUIDELINES.md as a structured transcription of the user-provided requirements; root overview and documentation navigation now link to the materials. Existing code/, campaigns/, results/, official input paths and frozen submission/ contents were preserved.

Report numbers were checked against best_model.json and owning summary/monthly/audit files. Report clearly identifies the inspected February–September backtests as developmental, distinguishes the original champion from its later RevIN trials and the older 363-day inference bundle, and describes pending final-refit/calibration and offline-delivery work. Three pages were rendered with Poppler and visually reviewed. The report builder uses optional requirements-report.txt and does no model execution. A Python-only file/checklist audit confirms all 16 frozen file hashes, all preparation files, report-source/PDF identity and visual-review receipt. It deliberately exits 1 for the open final-handoff items. No model fitting, inference, external upload, video recording or final event handoff occurred during preparation.

Remaining items: member names later; champion all-history refit per docs/SUBMISSION_REFIT_PLAN.md; actual fitted-artifact delivery; clean-environment offline inference test; final report/team review; repository license/data/third-party notice decisions; real <=120-second video and SwissTransfer link; final authorized publication/handoff. Working deadline is Sunday 18 October 2026 11:00 Europe/Zurich, resolved from the supplied Sunday guidance and subject to organizer updates. Authorized Git target remains neural-wave/ch-PHDs-but-not-for-long, branch el_bicho/hackaton-research; organizer generic NW_26 naming requires reconciliation if it conflicts, not an invented alternate target.

Verification: new Python sources compiled, new internal Markdown links resolved, no README missing links, PDF exactly three pages, frozen hashes unchanged. Receipts under results/submission_organization_20261017/. The complete previous startup summary was archived verbatim to memory/records/20261017_startup_before_submission_organization.md; unrelated interpretability and illustrative-band work remains preserved. Live git ls-remote showed the authorized branch at a75646dcafacce44b7c02a26faea055548aab6dd before preparation publication; publication is handled separately and must be verified.
