#!/usr/bin/env python3
"""Paired-seed leave-one-feature-group-out ablations for zonal CNNs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import validate_aug_sep as core

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_prices_15min.csv"
MONTH_RUN = ROOT / "experiments" / "month_difficulty_lomo_aug_sep"
BASE_RUN = ROOT / "experiments" / "aug_sep_per_zone_fresh"
OUTPUT = ROOT / "experiments" / "feature_ablation_per_zone"
VAL_START, VAL_END = "2026-08-01", "2026-09-30"
MONTH_STRENGTH = 0.5
SEED_OFFSETS = (0, 1, 2)
MAX_EPOCHS, PATIENCE = 150, 16

PROFILE_FEATURES = {
    "normalized_level": 0,
    "day_centered_level": 1,
    "hourly_slope": 2,
    "hourly_curvature": 3,
    "boundary_indicator": 4,
}
CALENDAR_FEATURES = ("hour_embedding", "weekday_embedding", "month_embedding")
ABLATIONS = {"full": {"profile": tuple(PROFILE_FEATURES), "calendar": CALENDAR_FEATURES}}
for name in PROFILE_FEATURES:
    ABLATIONS[f"without_{name}"] = {
        "profile": tuple(k for k in PROFILE_FEATURES if k != name),
        "calendar": CALENDAR_FEATURES,
    }
for name in CALENDAR_FEATURES:
    ABLATIONS[f"without_{name}"] = {
        "profile": tuple(PROFILE_FEATURES),
        "calendar": tuple(k for k in CALENDAR_FEATURES if k != name),
    }
ABLATIONS["without_normalized_level_and_month_embedding"] = {
    "profile": tuple(k for k in PROFILE_FEATURES if k != "normalized_level"),
    "calendar": tuple(k for k in CALENDAR_FEATURES if k != "month_embedding"),
}


class ZonalAblationCNN(nn.Module):
    def __init__(self, config, profile_count, calendar_features):
        super().__init__()
        self.include_hour = "hour_embedding" in calendar_features
        self.include_weekday = "weekday_embedding" in calendar_features
        self.include_month = "month_embedding" in calendar_features
        if self.include_hour:
            self.hour = nn.Embedding(24, 6)
        if self.include_weekday:
            self.weekday = nn.Embedding(7, 4)
        if self.include_month:
            self.month = nn.Embedding(12, 4)
        channels = profile_count + (6 if self.include_hour else 0) \
            + (4 if self.include_weekday else 0) + (4 if self.include_month else 0)
        blocks = []
        for _ in range(config["depth"]):
            blocks.extend([nn.Conv1d(channels, config["width"], 3, padding=1),
                           nn.GELU(), nn.Dropout(config["dropout"])])
            channels = config["width"]
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(config["width"], 4, 1)

    def forward(self, profile, weekday, month):
        n = len(profile)
        values = [profile]
        if self.include_hour:
            hour_ids = torch.arange(24, device=profile.device)
            values.append(self.hour(hour_ids)[None].expand(n, -1, -1))
        cal = []
        if self.include_weekday:
            cal.append(self.weekday(weekday))
        if self.include_month:
            cal.append(self.month(month))
        if cal:
            date = torch.cat(cal, dim=1)
            values.append(date[:, None, :].expand(-1, 24, -1))
        sequence = torch.cat(values, dim=-1)
        return self.head(self.encoder(sequence.transpose(1, 2))).transpose(1, 2)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qweighted_mae(actual, predicted):
    weights = np.asarray(core.Q_WEIGHT, dtype=np.float64).reshape(1, 1, 4)
    return float((np.abs(predicted - actual) * weights).mean() / weights.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=tuple(ABLATIONS),
                        help="run only this follow-up variant and compare it with saved full-feature seeds")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    keys, hourly, target = core.read_profiles(SOURCE)
    weekday, month_ids = core.calendar(keys)
    leaderboard = read_rows(BASE_RUN / "leaderboard.csv")
    monthly_weights = read_rows(MONTH_RUN / "monthly_training_weights.csv")
    configs, trial_ids, weight_lookup = {}, {}, {}
    for zone in core.ZONES:
        # Use the exact CNN configurations and selected trial IDs used in the
        # established month-weighted MAE experiment.
        run_info = json.loads((MONTH_RUN / "run_info.json").read_text())
        configs[zone] = run_info["final_cnn_configs"][zone]
        # Match the selected CNN row exactly to retain its seed/trial identity.
        trial_ids[zone] = next(int(r["trial"]) for r in leaderboard
                                if r["zone"] == zone and r["model"] == "cnn"
                                and json.loads(r["config"]) == configs[zone])
        for row in monthly_weights:
            if (row["training_loss"] == "mae" and row["zone"] == zone
                    and row["used_in_final_training"].lower() == "true"):
                weight_lookup[(zone, row["month"])] = float(row["final_training_weight"])

    detail_rows = []
    start_mask = np.asarray([day < VAL_START for day, _ in keys])
    val_mask = np.asarray([VAL_START <= day <= VAL_END for day, _ in keys])
    qweights = torch.tensor(core.Q_WEIGHT, dtype=torch.float32, device=device).view(1, 1, 4)

    selected_ablations = ({args.ablation: ABLATIONS[args.ablation]} if args.ablation
                          else ABLATIONS)
    for zi, zone in enumerate(core.ZONES):
        zone_ids = np.asarray([i for i, (_, z) in enumerate(keys) if z == zone])
        train_ids = zone_ids[start_mask[zone_ids]]
        val_ids = zone_ids[val_mask[zone_ids]]
        hmean, hscale = float(hourly[train_ids].mean()), float(hourly[train_ids].std())
        all_profile = core.profile_features(hourly, hmean, hscale)
        residual_scale = float(np.mean(np.abs(target[train_ids] - hourly[train_ids, :, None])))
        config = configs[zone]
        sample_weight = np.ones(len(keys), dtype=np.float32)
        for i in train_ids:
            sample_weight[i] = weight_lookup[(zone, keys[i][0][:7])]
        tx_all = torch.as_tensor(all_profile, dtype=torch.float32, device=device)
        ty_all = torch.as_tensor(target, dtype=torch.float32, device=device)
        wd_all = torch.as_tensor(weekday, dtype=torch.long, device=device)
        mo_all = torch.as_tensor(month_ids, dtype=torch.long, device=device)
        weight_all = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
        vx = all_profile[val_ids]
        vy = target[val_ids]
        vw = torch.as_tensor(weekday[val_ids], dtype=torch.long, device=device)
        vm = torch.as_tensor(month_ids[val_ids], dtype=torch.long, device=device)
        vx_t = torch.as_tensor(vx, dtype=torch.float32, device=device)

        for ablation_name, features in selected_ablations.items():
            selected = [PROFILE_FEATURES[name] for name in features["profile"]]
            profile_count = len(selected)
            for seed_idx, offset in enumerate(SEED_OFFSETS):
                seed = core.SEED + zi * 10000 + 100 + trial_ids[zone] + offset * 1_000_003
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32 - 1))
                random.seed(seed)
                model = ZonalAblationCNN(config, profile_count, features["calendar"]).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                              weight_decay=config["wd"])
                train_profile = tx_all[train_ids][:, :, selected]
                train_target = ty_all[train_ids]
                train_wd = wd_all[train_ids]
                train_mo = mo_all[train_ids]
                train_month_weight = weight_all[train_ids]
                loader = DataLoader(TensorDataset(train_profile, train_target, train_wd,
                                                   train_mo, train_month_weight),
                                    batch_size=config["batch"], shuffle=True)
                val_profile = vx_t[:, :, selected]
                best_score, best_state, best_epoch, waiting = float("inf"), None, 0, 0
                started = time.perf_counter()
                for epoch in range(1, MAX_EPOCHS + 1):
                    model.train()
                    for bx, by, bw, bm, bmonth in loader:
                        optimizer.zero_grad(set_to_none=True)
                        raw = model(bx, bw, bm) * residual_scale
                        pred = core.center(raw, by.mean(dim=-1))
                        loss = ((pred - by).abs() * qweights * bmonth[:, None, None]).mean()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        raw = model(val_profile, vw, vm).cpu().numpy().astype(np.float64)
                    raw *= residual_scale
                    pred = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
                    val_mae = float(np.abs(pred - vy).mean())
                    if val_mae < best_score - 1e-6:
                        best_score, best_epoch, waiting = val_mae, epoch, 0
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
                    else:
                        waiting += 1
                        if waiting >= PATIENCE:
                            break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    raw = model(val_profile, vw, vm).cpu().numpy().astype(np.float64) * residual_scale
                pred = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
                err = np.abs(pred - vy)
                row = {"ablation": ablation_name, "zone": zone, "seed_index": seed_idx,
                       "seed": seed, "included_profile_features": ";".join(features["profile"]),
                       "included_calendar_embeddings": ";".join(features["calendar"]),
                       "parameter_count": sum(p.numel() for p in model.parameters()
                                              if p.requires_grad),
                       "mae": float(err.mean()), "qweighted_mae": qweighted_mae(vy, pred),
                       "q1_mae": float(err[:, :, 0].mean()),
                       "q2_mae": float(err[:, :, 1].mean()),
                       "q3_mae": float(err[:, :, 2].mean()),
                       "q4_mae": float(err[:, :, 3].mean()),
                       "best_epoch": best_epoch, "seconds": time.perf_counter() - started}
                detail_rows.append(row)
                print(f"{ablation_name} {zone} seed{seed_idx}: MAE={row['mae']:.4f}; "
                      f"qMAE={row['qweighted_mae']:.4f}; epoch={best_epoch}", flush=True)

    if args.ablation:
        followup_path = OUTPUT / f"confirmatory_{args.ablation}.csv"
        write_rows(followup_path, detail_rows)
        full_rows = [r for r in read_rows(OUTPUT / "ablation_runs.csv") if r["ablation"] == "full"]
        paired = []
        for row in detail_rows:
            ref = next(r for r in full_rows if r["zone"] == row["zone"]
                       and int(r["seed_index"]) == row["seed_index"])
            paired.append({"zone": row["zone"], "seed_index": row["seed_index"],
                           "full_mae": float(ref["mae"]), "reduced_mae": float(row["mae"]),
                           "delta_mae": float(row["mae"]) - float(ref["mae"]),
                           "full_qweighted_mae": float(ref["qweighted_mae"]),
                           "reduced_qweighted_mae": float(row["qweighted_mae"]),
                           "delta_qweighted_mae": float(row["qweighted_mae"])
                               - float(ref["qweighted_mae"])})
        write_rows(OUTPUT / f"confirmatory_{args.ablation}_paired_deltas.csv", paired)
        print(f"Saved follow-up ablation to {followup_path}", flush=True)
        return

    write_rows(OUTPUT / "ablation_runs.csv", detail_rows)
    summary = []
    full = {}
    for zone in core.ZONES:
        vals = [r["mae"] for r in detail_rows if r["ablation"] == "full" and r["zone"] == zone]
        full[zone] = float(np.mean(vals))
    for ablation_name in ABLATIONS:
        for zone in core.ZONES:
            subset = [r for r in detail_rows if r["ablation"] == ablation_name and r["zone"] == zone]
            maes = np.asarray([r["mae"] for r in subset])
            qmaes = np.asarray([r["qweighted_mae"] for r in subset])
            summary.append({"ablation": ablation_name, "zone": zone,
                            "mae_mean": float(maes.mean()), "mae_std": float(maes.std(ddof=1)),
                            "qweighted_mae_mean": float(qmaes.mean()),
                            "qweighted_mae_std": float(qmaes.std(ddof=1)),
                            "delta_mae_vs_full": float(maes.mean() - full[zone]),
                            "seed_count": len(maes)})
        overall = np.mean([r["mae_mean"] for r in summary if r["ablation"] == ablation_name])
        summary.append({"ablation": ablation_name, "zone": "MEAN_ZONES",
                        "mae_mean": float(overall), "mae_std": "",
                        "qweighted_mae_mean": float(np.mean([
                            r["qweighted_mae_mean"] for r in summary
                            if r["ablation"] == ablation_name and r["zone"] in core.ZONES])),
                        "qweighted_mae_std": "",
                        "delta_mae_vs_full": float(overall - np.mean(list(full.values()))),
                        "seed_count": len(SEED_OFFSETS)})
    write_rows(OUTPUT / "ablation_summary.csv", summary)
    info = {"device": str(device), "validation_period": [VAL_START, VAL_END],
            "training_period": ["2025-10-01", "2026-07-31"],
            "zones": list(core.ZONES), "loss": "MAE, with quarter weights 1.5,1,1,1.5",
            "month_weight_strength": MONTH_STRENGTH,
            "month_weights_source": str((MONTH_RUN / "monthly_training_weights.csv").relative_to(ROOT)),
            "models": "one CNN per zone; each zone uses its previously selected configuration",
            "configurations": configs, "selected_trial_ids": trial_ids,
            "seed_offsets": list(SEED_OFFSETS), "feature_groups": ABLATIONS,
            "selection_metric": "plain validation MAE; lower is better",
            "ablation_definition": "remove one feature/channel group at a time; keep all other inputs and training choices fixed",
            "run_count": len(detail_rows)}
    (OUTPUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Saved {len(detail_rows)} fits on {device} to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
