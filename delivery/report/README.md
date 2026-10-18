# Three page report draft

[`report.pdf`](report.pdf) is the three-page jury report draft. [`report.md`](report.md) is its editable source. The chart is built from the saved champion monthly scores, with the aggregate ratio checked against `best_model.json`.

From the repository root, install the optional authoring tools and rebuild:

```sh
python -m pip install -r requirements-report.txt
python run.py campaigns/submission_materials_20261017/build_report.py
```

The builder performs no model fitting or inference. It checks that the output has exactly three pages and writes a receipt to [`results/submission_organization_20261017/report_build.json`](../../results/submission_organization_20261017/report_build.json). Rebuilding resets visual review to pending: render and inspect all pages again before updating that receipt. Team review is a separate pending item in the submission checklist.

Before final export, add the member list, review the AI usage disclosure, and replace the pending deployment paragraph with the actual audited final-model status. Keep the developmental-results qualification and distinguish the champion from the older preserved deployment recipe.
