#!/usr/bin/env python3
"""Per-zone MPS HPO on August–September 2026; reads only official training CSV.

Model inputs are built from the 24 hourly prices of that delivery day and the
calendar. No quarter-hour values, October data, or past-day prices enter the
features. The quarter-hour values are training labels / validation targets.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_prices_15min.csv"
OUTPUT = ROOT / "experiments" / "aug_sep_per_zone_fresh"
ZONES = ("ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3")
SEED = 20261017
Q_WEIGHT = (1.5, 1.0, 1.0, 1.5)


def read_profiles(path):
    bins = defaultdict(lambda: np.full((24, 4), np.nan, dtype=np.float64))
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bins[(r["delivery_date"], r["market_id"])][int(r["hour_index"]) - 1,
                                                        int(r["quarter_index"]) - 1] = float(r["price_eur_mwh"])
    keys, hourly, quarter = [], [], []
    for key, y in sorted(bins.items()):
        if not np.isfinite(y).all():
            continue
        keys.append(key)
        quarter.append(y)
        hourly.append(y.mean(axis=1))
    return keys, np.asarray(hourly), np.asarray(quarter)


def calendar(keys):
    # Weekday is Monday=0. Month is one-hot encoded below.
    days = [np.datetime64(d, "D") for d, _ in keys]
    weekday = np.asarray([(int(d.astype(int)) + 3) % 7 for d in days], dtype=np.int64)
    month = np.asarray([int(str(d)[5:7]) - 1 for d in days], dtype=np.int64)
    return weekday, month


def profile_features(hourly, mean, scale):
    z = (hourly - mean) / scale
    daily_centered = (hourly - hourly.mean(axis=1, keepdims=True)) / scale
    slope = np.gradient(z, axis=1)
    curvature = np.gradient(slope, axis=1)
    # Explicitly expose daily edges and neighboring-hour context to edge quarters.
    boundary = np.zeros_like(z)
    boundary[:, 0] = 1.0
    boundary[:, -1] = 1.0
    boundary[:, 1] = -1.0
    boundary[:, -2] = -1.0
    return np.stack([z, daily_centered, slope, curvature, boundary], axis=-1).astype(np.float32)


class PerZoneNet(nn.Module):
    def __init__(self, kind, width, depth, dropout, feature_channels=5):
        super().__init__()
        self.kind = kind
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)
        self.hour = nn.Embedding(24, 6)
        if kind == "mlp":
            size = 24 * feature_channels + 8
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout)])
                size = width
            layers.append(nn.Linear(size, 96))
            self.net = nn.Sequential(*layers)
        elif kind in ("cnn", "tcn"):
            blocks = []
            channels = feature_channels + 6 + 8
            for layer in range(depth):
                dilation = 1 if kind == "cnn" else 2 ** layer
                conv = nn.Conv1d(channels, width, 3, padding=dilation, dilation=dilation)
                blocks.extend([conv, nn.GELU(), nn.Dropout(dropout)])
                channels = width
            self.encoder = nn.Sequential(*blocks)
            self.head = nn.Conv1d(width, 4, 1)
        elif kind == "gru":
            self.encoder = nn.GRU(feature_channels + 6 + 8, width, num_layers=depth, batch_first=True,
                                  dropout=dropout if depth > 1 else 0.0)
            self.head = nn.Linear(width, 4)

    def forward(self, profile, weekday, month):
        n = len(profile)
        cal = torch.cat([self.weekday(weekday), self.month(month)], dim=1)
        hour_ids = torch.arange(24, device=profile.device)
        hour = self.hour(hour_ids)[None].expand(n, -1, -1)
        seq = torch.cat([profile, hour, cal[:, None, :].expand(-1, 24, -1)], dim=-1)
        if self.kind == "mlp":
            return self.net(torch.cat([profile.flatten(1), cal], dim=1)).reshape(n, 24, 4)
        if self.kind in ("cnn", "tcn"):
            return self.head(self.encoder(seq.transpose(1, 2))).transpose(1, 2)
        encoded, _ = self.encoder(seq)
        return self.head(encoded)


def center(raw, hourly):
    return hourly[..., None] + raw - raw.mean(dim=-1, keepdim=True)


def metrics(y, hourly, raw):
    pred = hourly[..., None] + raw - raw.mean(axis=-1, keepdims=True)
    err = np.abs(pred - y)
    return {"mae": float(err.mean()),
            "q1_mae": float(err[:, :, 0].mean()), "q2_mae": float(err[:, :, 1].mean()),
            "q3_mae": float(err[:, :, 2].mean()), "q4_mae": float(err[:, :, 3].mean()),
            "mean_constraint_max_abs": float(np.max(np.abs(pred.mean(axis=-1) - hourly)))}


def trial(kind, cfg, features, weekday, month, hourly, target, residual_scale, train_ids, val_ids, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = PerZoneNet(kind, cfg["width"], cfg["depth"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tx = torch.as_tensor(features[train_ids], dtype=torch.float32, device=device)
    ty = torch.as_tensor(target[train_ids], dtype=torch.float32, device=device)
    tw = torch.as_tensor(weekday[train_ids], dtype=torch.long, device=device)
    tm = torch.as_tensor(month[train_ids], dtype=torch.long, device=device)
    vx = torch.as_tensor(features[val_ids], dtype=torch.float32, device=device)
    vw = torch.as_tensor(weekday[val_ids], dtype=torch.long, device=device)
    vm = torch.as_tensor(month[val_ids], dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(tx, ty, tw, tm), batch_size=cfg["batch"], shuffle=True)
    weights = torch.tensor(Q_WEIGHT, dtype=torch.float32, device=device).view(1, 1, 4)
    best, best_epoch, best_state, waiting = float("inf"), 0, None, 0
    for epoch in range(1, 121):
        model.train()
        for bx, by, bw, bm in loader:
            opt.zero_grad(set_to_none=True)
            raw = model(bx, bw, bm) * residual_scale
            hourly_batch = by.mean(dim=-1)
            pred = center(raw, hourly_batch)
            loss = ((pred - by).abs() * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            raw = model(vx, vw, vm).cpu().numpy().astype(np.float64) * residual_scale
        val = metrics(target[val_ids], hourly[val_ids], raw)
        if val["mae"] < best - 1e-6:
            best, best_epoch, waiting = val["mae"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= 14:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(vx, vw, vm).cpu().numpy().astype(np.float64) * residual_scale
    return model, metrics(target[val_ids], hourly[val_ids], raw), best_epoch, raw


def search_configs():
    grid = list(itertools.product((32, 64, 96), (1, 2, 3), (0.0, 0.15),
                                  (32, 64), (1e-3, 3e-4), (1e-4, 1e-3)))
    by_model = {}
    for mi, kind in enumerate(("mlp", "cnn", "tcn", "gru")):
        rng = random.Random(SEED + mi)
        values = grid.copy()
        rng.shuffle(values)
        by_model[kind] = [{"width": w, "depth": d, "dropout": dr, "batch": b, "lr": lr, "wd": wd}
                          for w, d, dr, b, lr, wd in values[:8]]
    return by_model


def linear_baseline(hourly):
    xh = np.arange(24, dtype=np.float64) + 0.5
    xq = (np.arange(96, dtype=np.float64) + 0.5) / 4.0
    out = np.stack([np.interp(xq, xh, day) for day in hourly]).reshape(-1, 24, 4)
    out += (hourly - out.mean(axis=-1))[:, :, None]
    return out


def main():
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.set_num_threads(2)
    device = torch.device("mps")
    keys, hourly, target = read_profiles(SOURCE)
    train_mask = np.asarray([day < "2026-08-01" for day, _ in keys])
    val_mask = np.asarray(["2026-08-01" <= day <= "2026-09-30" for day, _ in keys])
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("Expected train data through 2026-09-30")
    weekday, month = calendar(keys)
    output_rows, by_zone, selected, pred_rows = [], {}, {}, []
    config_map = search_configs()
    for zi, zone in enumerate(ZONES):
        zone_ids = np.asarray([i for i, (_, z) in enumerate(keys) if z == zone])
        tr = zone_ids[train_mask[zone_ids]]
        va = zone_ids[val_mask[zone_ids]]
        hmean, hscale = float(hourly[tr].mean()), float(hourly[tr].std())
        features = profile_features(hourly, hmean, hscale)
        # A residual scale fitted on train dates only stabilizes all architectures.
        residual_scale = float(np.mean(np.abs(target[tr] - hourly[tr, :, None])))
        baseline = linear_baseline(hourly[va])
        by_zone[zone] = {"linear_interpolation": metrics(target[va], hourly[va],
                                                         baseline - hourly[va, :, None])}
        best = None
        for ki, kind in enumerate(("mlp", "cnn", "tcn", "gru")):
            for ti, cfg in enumerate(config_map[kind], 1):
                started = time.perf_counter()
                _, met, epoch, raw = trial(kind, cfg, features, weekday, month, hourly, target,
                                           residual_scale, tr, va, device,
                                           SEED + zi * 10000 + ki * 100 + ti)
                row = {"zone": zone, "model": kind, "trial": ti, **met,
                       "best_epoch": epoch, "seconds": time.perf_counter() - started,
                       "config": cfg}
                output_rows.append(row)
                print(f"{zone} {kind}_{ti:02d}: MAE={met['mae']:.5f}; "
                      f"Q1={met['q1_mae']:.3f} Q4={met['q4_mae']:.3f} (epoch {epoch})", flush=True)
                if best is None or met["mae"] < best[0]:
                    best = (met["mae"], kind, ti, cfg, epoch, raw, met)
        _, kind, ti, cfg, epoch, raw, met = best
        selected[zone] = {"model": kind, "trial": ti, "config": cfg,
                          "best_epoch": epoch, "aug_sep_metrics": met}
        by_zone[zone]["best_neural"] = met
        # Store validation predictions with the requested zone/date/hour/quarter keys.
        pred = hourly[va, :, None] + raw - raw.mean(axis=-1, keepdims=True)
        for j, idx in enumerate(va):
            day = keys[idx][0]
            for hi in range(24):
                for qi in range(4):
                    pred_rows.append([day, zone, hi + 1, qi + 1,
                                      float(target[idx, hi, qi]), float(pred[j, hi, qi])])
        del features

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["zone", "model", "trial", "mae", "q1_mae", "q2_mae", "q3_mae", "q4_mae",
                  "mean_constraint_max_abs", "best_epoch", "seconds", "config"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(output_rows, key=lambda x: (x["zone"], x["mae"])):
            w.writerow({**r, "config": json.dumps(r["config"], sort_keys=True)})
    with (OUTPUT / "best_model_validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["delivery_date", "market_id", "hour_index", "quarter_index", "actual_eur_mwh", "predicted_eur_mwh"])
        w.writerows(pred_rows)
    dates = sorted({day for day, _ in keys})
    info = {"validation_period": ["2026-08-01", "2026-09-30"],
            "train_period": [dates[0], "2026-07-31"], "zones": list(ZONES),
            "per_zone_models": selected, "per_zone_metrics": by_zone,
            "mean_mae_across_four_zones": float(np.mean([by_zone[z]["best_neural"]["mae"] for z in ZONES])),
            "trial_count": len(output_rows), "trials_per_zone": 32, "device": str(device),
            "input_policy": "same-day 24-hour prices plus calendar; no quarter-hour features or lagged-day prices",
            "edge_feature_engineering": ["hourly slope", "hourly curvature", "day-centered level", "explicit first/last-hour edge channel"],
            "edge_weight_in_training_loss": {"q1": 1.5, "q2": 1.0, "q3": 1.0, "q4": 1.5},
            "october_data_opened": False,
            "mean_preserving_transform": "hourly + (raw - mean(raw over four quarters))"}
    (OUTPUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"mean_mae_across_four_zones": info["mean_mae_across_four_zones"],
                      "selected_models": selected, "baseline_and_best_metrics": by_zone,
                      "output": str(OUTPUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
