# Final submission checklist

Status on 17 October 2026. Checked items describe preparation completed in this repository; unchecked items still require action. See the [organizer guidance](../docs/reference/SUBMISSION_GUIDELINES.md) and [AET specification](../docs/reference/01_v02_AET_Neural_Wave_2026_challenge_EN.pdf).

## Code and model

- [x] Keep clear folders for shared code, studies, documentation, results and jury materials.
- [x] Provide a root README, pip dependency files and documented training/inference commands.
- [x] Preserve the existing frozen `submission/` files with `FREEZE_MANIFEST.json`.
- [x] Add the confirmed team name, all four full names and supplied roles to the README and credits.
- [ ] Implement and audit the current research champion's all-history refit on an authorized research server, following [the refit plan](../docs/SUBMISSION_REFIT_PLAN.md). Freeze configuration before fitting; resolve calibration transfer separately.
- [ ] Package the fitted artifact with all required source files, configuration, seeds and checksums. Verify it can be regenerated from the official input.
- [ ] Make the root quick start, jury run guide and report name the actual final artifact and recipe consistently.
- [ ] Verify installation with one pip command in a fresh supported environment; record Python/package versions and installation outcome.
- [ ] Execute final inference without network access on a small official-data fixture; record the command, environment, artifact hash and output audit. Any substantial inference belongs on a research server.
- [ ] Verify exact CSV columns, complete unique keys, four finite quarters per hour, correct zone/date/hour coverage and mean error <=0.0001 EUR/MWh.
- [ ] Verify history is strictly earlier than each evaluated date and inference performs no refitting, recalibration, selection or external lookup.

## Report and credits

- [x] Prepare an editable report draft covering approach, results, challenges and AI assistant usage.
- [x] Generate and visually inspect a three-page PDF draft with a monthly-results chart.
- [x] Identify official data, software dependencies and AI assistant use in the credits.
- [ ] Review the report as a team; add names, confirm AI usage details and align deployment claims with the final artifact.
- [ ] Confirm the final exported report remains at most three pages, including references.
- [ ] Choose and add the team's repository license; confirm handling of organizer-provided data and retained third-party notices for the post-event public repository.

## Video and presentation

- [x] Prepare a two-minute narration and visual plan using verified results.
- [ ] Record the video with team narration and a real model demo if ready; keep duration <=120 seconds.
- [ ] Watch the export once for readable visuals, audible speech, correct results and correct final model identity.
- [ ] Upload the video to SwissTransfer and record the working download link; verify its availability through judging.
- [x] Prepare a timed five-minute presentation outline.
- [ ] Prepare/rehearse the final presentation visuals if needed for the winners' presentation.

## Publication and handoff

- [ ] Resolve any repository-name/branch discrepancy with the organizers. Current authorized target is `neural-wave/ch-PHDs-but-not-for-long`, branch `el_bicho/hackaton-research`.
- [ ] Publish the final code and report to that target, and verify the remote contains the intended files.
- [ ] Verify judges can obtain the fitted artifact; ignored model binaries will not appear in a normal Git clone.
- [ ] Submit only one package per team, with the code/report location and SwissTransfer video link.
- [ ] Complete handoff by the working deadline, Sunday 18 October 2026 at 11:00 Europe/Zurich, subject to organizer updates.
- [ ] Save final handoff evidence, clean-environment audit and immutable artifact hashes in `results/`.
