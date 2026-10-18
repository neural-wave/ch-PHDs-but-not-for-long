---
title: Completion evidence audited; explicit raw-interpolation comparison added; local
  checksum pending
status: historical
kind: audit
recorded_at: '2026-10-17T16:02:34.369342+00:00'
sources:
- results/minimize_expanding_20261017/COMPLETION_AUDIT.md
- results/minimize_expanding_20261017/INTERPRETATION.md
- results/minimize_expanding_20261017/aggregate_comparison.csv
- results/minimize_expanding_20261017/alternate_interpolation_monthly_scores.csv
- campaigns/minimize_expanding_20261017/aggregate_baselines.py
experiment_id: minimize_expanding_20261017
supersedes: []
status_updated_at: '2026-10-17T16:07:52.728227+00:00'
---

17 October 2026 around18:02Zurich: previous goal turn was a verified wait for specific backup Nightwatch8c78bfcc73e7ff61f0b958cd, not a blocker. This turn adds substantive completion evidence and fills the protocol's explicitly alternate raw-interpolation aggregate reporting. Scientific result remains fully audited: primary pre-test selector0.7088921710561056, pooledMAE4.107397336721399; expanding eight-fold training3–10months. No models, selections or frozen training sources changed.

Added aggregate_baselines.py (stdlib, about1000 small CSV rows, no fitting/inference). Reconstructs every recipe/month from four identical-size zone subsets by aggregating absolute errors before taking ratios. Checks all primary monthly/aggregate MAEs against saved audited tables, all denominators positive, eight folds/241days. Output alternate_interpolation_monthly_scores.csv and aggregate_comparison.csv saved locally/remotely. Selector distinct pooled corrected-interpolation ratio0.7063674147285953; explicitly alternate equal-month uncorrected-linear-interpolation ratio0.6727719501477243 and pooled raw ratio0.6698521224649152. Pooled corrected/raw baseline MAEs5.814817120774983/6.131797151895316. The primary metric remains mean-corrected interpolation0.708892; never silently substitute raw interpolation or pooled ratios.

COMPLETION_AUDIT.md now maps every requested/protocol requirement to owning sources and actual verification: exact expanding chronology, source and official-data identity, causal perturbation/truncation, fresh training with optimizer indices below nt, selection/calibration separation, training-only weights/knots, frozen menus, conservation, eleven quantiles/intervals, matched monthly/zone/aggregate errors, pre-test versus observed grid winner, remote tracking and reviewed artifacts. Re-read current training source confirms init y[:nt], target arrays only training+selection and optimizer randperm(nt), independent calibration slice ns:ns+nc. Source hashes already independently matched the full execution manifest. Scientific requirements verified; only final local recovery checksum and final project-status update remain pending. No claim of global mathematical optimality or blind benchmark.

Backup still live: latest observed Nightwatchheartbeat1792252833.2056642,statusrunning,exitcodenull,heartbeat_expectedtrue/180secondtimeout; local .partial file grew to1.5GiB around18:01Zurich. Local ps is sandbox-blocked, so no PID-liveness claim from that command; actual job API heartbeats and growing transfer establish ongoing work. Scientific Nightwatch0c174584a3b9236d9f21ffd4 remainscompletedexit0. Preserve the same backup handle; do not restart because a poll times out. Next verify backup_verification.json and actual terminal exit, update audit/interpretation/startup and finish goal only after that evidence.
