# PhDs but not for long — submission materials

This folder collects the jury materials and the final handoff checklist for the AET challenge. **Preparation is in progress; this is not a completed submission.** The frozen executable bundle remains in [`submission/`](../submission/README.md).

## Deliverables

| Requirement | Location | Current status |
| --- | --- | --- |
| Repository overview and team | [Root README](../README.md) | PhDs but not for long; all four members and roles recorded |
| Training and inference code | [`code/`](../docs/CODE_INDEX.md), [`campaigns/`](../campaigns/), [`submission/`](../submission/README.md) | Existing research code and older frozen bundle preserved |
| Dependencies and execution | [Jury run guide](RUN_FOR_JURY.md) | Champion CPU runtime, pinned dependencies and remote clean-install/CSV audit verified; see guide |
| Report, maximum 3 pages | [PDF draft](report/report.pdf), [editable source](report/report.md) | Three-page research report draft; final model and team details pending |
| Video, maximum 2 minutes | [Recording plan and script](video/SCRIPT.md) | Script prepared; recording and SwissTransfer link pending |
| Presentation, maximum 5 minutes | [Presentation outline](presentation/OUTLINE.md) | Outline prepared for the optional winners' presentation |
| Credits and open-source preparation | [Credits](CREDITS.md) | Data/software/AI usage recorded; team license choice pending |
| Final review | [Checklist](CHECKLIST.md), [machine-readable status](submission_status.json) | Open items explicitly tracked |

## Repository organization

```text
README.md                  project overview, team, quick start
requirements*.txt          research dependency files
run.py                     repository script launcher
code/                      shared models, training, evaluation, utilities
campaigns/                 individual research studies and reproduction scripts
docs/                      execution guides, protocols, original challenge files
results/                   metrics, figures, audits and provenance
submission/                preserved frozen executable bundle and its manifest
delivery/                  report, video plan, presentation and final checklist
train_prices_15min.csv     official input, kept at its established path
memory/records/            dated research and operational records
```

The organizer's `src/`, `data/`, and `notebooks/` names are examples. This repository uses the established names above; scripts and frozen checksums depend on their current paths. Notebooks are not required to run the solution.

## Final handoff

1. Finish the [open checklist](CHECKLIST.md), including the final all-history model and clean-environment audit.
2. Confirm the final delivery model, report, video and run instructions describe the same recipe. Preserve the current `submission/` bundle until an audited replacement is deliberately made.
3. Publish the final code and report to the authorized repository, [neural-wave/ch-PHDs-but-not-for-long](https://github.com/neural-wave/ch-PHDs-but-not-for-long), branch `el_bicho/hackaton-research`. The organizer's generic `NW_26_<Company Name>_<Team Name>` naming guidance does not establish a different authorized target; resolve any discrepancy with the organizers before handoff.
4. Deliver the actual fitted artifact along with its complete code, dependency files and checksums through the organizer-approved channel. `*.pkl` and large archives are ignored by Git; a clone alone is not an executable model delivery.
5. Record the video, upload it to SwissTransfer, and put the working link in `submission_status.json`. Keep the final code/report and video as one team submission.
6. Use Sunday **18 October 2026, 11:00 Europe/Zurich** as the working deadline from the supplied event guidance; confirm any organizer update before the final handoff.

Run the preparation audit from the root:

```sh
python run.py check_submission --output results/submission_organization_20261017/readiness.json
```

An incomplete submission returns exit code 1 and lists the missing items. This is a packaging/status audit, not a clean-environment model execution test.
