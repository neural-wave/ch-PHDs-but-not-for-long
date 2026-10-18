#!/usr/bin/env python3
"""Refit zonal monthly-difficulty weights without a month embedding.

The script scores all 12 months with leave-one-month-out CNNs trained with
MAE and MSE, derives leakage-safe October--July weights, and evaluates the
selected reduced-feature zonal CNNs under both the old and revised weights.
"""
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
from feature_ablation_per_zone import ZonalAblationCNN

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train_prices_15min.csv"
OLD_MONTH_RUN = ROOT / "experiments" / "month_difficulty_lomo_aug_sep"
SELECTED_RUN = ROOT / "experiments" / "selected_reduced_features_per_zone"
OUTPUT = ROOT / "experiments" / "month_difficulty_no_month_embedding_aug_sep"

VAL_START, VAL_END = "2026-08-01", "2026-09-30"
MONTH_STRENGTH = 0.5
MAX_EPOCHS, PATIENCE = 150, 16
OBJECTIVES = ("mae", "mse")
SCORER_CONFIG = {"width": 64, "depth": 2, "dropout": 0.15,
                 "batch": 64, "lr": 3e-4, "wd": 1e-4}
PROFILE_FEATURE_NAMES = ["day_centered_level", "hourly_slope",
                         "hourly_curvature", "boundary_indicator"]
CALENDAR_FEATURE_NAMES = ["hour_embedding", "weekday_embedding"]
PROFILE_CHANNELS = [1, 2, 3, 4]
SEED_OFFSETS = {"ZONE_0": 4, "ZONE_1": 7, "ZONE_2": 6, "ZONE_3": 8}


class NoMonthScorerCNN(nn.Module):
    """The previous scorer CNN with its month embedding removed."""

    def __init__(self, config: dict, feature_channels: int = 5):
        super().__init__()
        self.hour = nn.Embedding(24, 6)
        self.weekday = nn.Embedding(7, 4)
        channels = feature_channels + 6 + 4
        blocks = []
        for _ in range(config["depth"]):
            blocks.extend([nn.Conv1d(channels, config["width"], 3, padding=1),
                           nn.GELU(), nn.Dropout(config["dropout"])])
            channels = config["width"]
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Conv1d(config["width"], 4, 1)

    def forward(self, profile: torch.Tensor, weekday: torch.Tensor) -> torch.Tensor:
        n = len(profile)
        hour_ids = torch.arange(24, device=profile.device)
        hour = self.hour(hour_ids)[None].expand(n, -1, -1)
        weekday_embedding = self.weekday(weekday)[:, None, :].expand(-1, 24, -1)
        sequence = torch.cat([profile, hour, weekday_embedding], dim=-1)
        return self.head(self.encoder(sequence.transpose(1, 2))).transpose(1, 2)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def month_of(day: str) -> str:
    return day[:7]


def qweighted_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    weights = np.asarray(core.Q_WEIGHT, dtype=np.float64).reshape(1, 1, 4)
    return float((np.abs(predicted - actual) * weights).mean() / weights.mean())


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(predicted - actual)
    return {
        "mae": float(error.mean()),
        "qweighted_mae": qweighted_mae(actual, predicted),
        **{f"q{q + 1}_mae": float(error[:, :, q].mean()) for q in range(4)},
    }


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def assert_no_month_embedding(model: nn.Module) -> None:
    names = [name for name, _ in model.named_parameters()]
    if any("month" in name.lower() for name in names):
        raise AssertionError(f"Month embedding unexpectedly present: {names}")


def fit_scorer(objective: str, seed: int, config: dict,
               profile: np.ndarray, target: np.ndarray, weekday: np.ndarray,
               hourly: np.ndarray, train_ids: np.ndarray, val_ids: np.ndarray,
               device: torch.device) -> tuple[np.ndarray, dict, int, int]:
    seed_everything(seed)
    model = NoMonthScorerCNN(config, profile.shape[-1]).to(device)
    assert_no_month_embedding(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["wd"])
    x = torch.as_tensor(profile[train_ids], dtype=torch.float32, device=device)
    y = torch.as_tensor(target[train_ids], dtype=torch.float32, device=device)
    wd = torch.as_tensor(weekday[train_ids], dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y, wd), batch_size=config["batch"], shuffle=True)
    target_scale = float(np.mean(np.abs(target[train_ids] - hourly[train_ids, :, None])))
    qweights = torch.tensor(core.Q_WEIGHT, dtype=torch.float32, device=device).view(1, 1, 4)
    vx = torch.as_tensor(profile[val_ids], dtype=torch.float32, device=device)
    vw = torch.as_tensor(weekday[val_ids], dtype=torch.long, device=device)
    best_score, best_state, best_epoch, waiting = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            raw = model(bx, bw) * target_scale
            pred = core.center(raw, by.mean(dim=-1))
            err = pred - by
            point_loss = err.abs() if objective == "mae" else err.square()
            loss = (point_loss * qweights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            raw = model(vx, vw).cpu().numpy().astype(np.float64) * target_scale
        predicted = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
        score = qweighted_mae(target[val_ids], predicted)
        if score < best_score - 1e-6:
            best_score, best_epoch, waiting = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Scorer did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(vx, vw).cpu().numpy().astype(np.float64) * target_scale
    predicted = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
    return predicted, metrics(target[val_ids], predicted), best_epoch, sum(
        p.numel() for p in model.parameters() if p.requires_grad)


def fit_selected(objective: str, weighting: str, zone: str, seed: int,
                 config: dict, sample_weights: np.ndarray, features: np.ndarray,
                 hourly: np.ndarray, target: np.ndarray, weekday: np.ndarray,
                 month_ids: np.ndarray, train_ids: np.ndarray, val_ids: np.ndarray,
                 device: torch.device) -> tuple[np.ndarray, dict, int, int, dict]:
    seed_everything(seed)
    model = ZonalAblationCNN(config, len(PROFILE_CHANNELS), CALENDAR_FEATURE_NAMES).to(device)
    assert_no_month_embedding(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["wd"])
    residual_scale = float(np.mean(np.abs(target[train_ids] - hourly[train_ids, :, None])))
    x = torch.as_tensor(features[train_ids], dtype=torch.float32, device=device)
    y = torch.as_tensor(target[train_ids], dtype=torch.float32, device=device)
    wd = torch.as_tensor(weekday[train_ids], dtype=torch.long, device=device)
    mo = torch.as_tensor(month_ids[train_ids], dtype=torch.long, device=device)
    mw = torch.as_tensor(sample_weights[train_ids], dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(x, y, wd, mo, mw), batch_size=config["batch"], shuffle=True)
    qweights = torch.tensor(core.Q_WEIGHT, dtype=torch.float32, device=device).view(1, 1, 4)
    vx = torch.as_tensor(features[val_ids], dtype=torch.float32, device=device)
    vw = torch.as_tensor(weekday[val_ids], dtype=torch.long, device=device)
    vm = torch.as_tensor(month_ids[val_ids], dtype=torch.long, device=device)
    best_score, best_state, best_epoch, waiting = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by, bw, bm, bmonth in loader:
            optimizer.zero_grad(set_to_none=True)
            raw = model(bx, bw, bm) * residual_scale
            pred = core.center(raw, by.mean(dim=-1))
            err = pred - by
            point_loss = err.abs() if objective == "mae" else err.square()
            loss = (point_loss * qweights * bmonth[:, None, None]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            raw = model(vx, vw, vm).cpu().numpy().astype(np.float64) * residual_scale
        predicted = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
        score = float(np.abs(predicted - target[val_ids]).mean())
        if score < best_score - 1e-6:
            best_score, best_epoch, waiting = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"Selected model did not produce a checkpoint for {zone}/{objective}/{weighting}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(vx, vw, vm).cpu().numpy().astype(np.float64) * residual_scale
    predicted = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
    checkpoint = {
        "state_dict": best_state,
        "zone": zone,
        "objective": objective,
        "weighting": weighting,
        "config": config,
        "profile_features": PROFILE_FEATURE_NAMES,
        "calendar_features": CALENDAR_FEATURE_NAMES,
        "month_embedding": False,
        "quarter_weights": list(core.Q_WEIGHT),
        "month_weight_strength": MONTH_STRENGTH,
        "month_weights": {},  # Filled for the saved refit-weight checkpoints.
        "residual_scale_train": residual_scale,
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return predicted, metrics(target[val_ids], predicted), best_epoch, checkpoint["parameter_count"], checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-mps", action="store_true",
                        help="fail if this process cannot access Apple MPS")
    args = parser.parse_args()
    torch.set_num_threads(2)
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if args.require_mps and not mps_available:
        raise RuntimeError("MPS was required but is unavailable in this process")
    device = torch.device("mps" if mps_available else "cpu")

    keys, hourly, target = core.read_profiles(SOURCE)
    weekday, month_ids = core.calendar(keys)
    months = sorted({month_of(day) for day, _ in keys})
    if len(months) != 12:
        raise RuntimeError(f"Expected 12 distinct months, found {months}")
    train_mask = np.asarray([day < VAL_START for day, _ in keys])
    val_mask = np.asarray([VAL_START <= day <= VAL_END for day, _ in keys])
    train_months = sorted({month_of(day) for (day, _), keep in zip(keys, train_mask) if keep})
    if len(train_months) != 10 or months != train_months + ["2026-08", "2026-09"]:
        raise RuntimeError(f"Unexpected month split: all={months}, train={train_months}")

    old_rows = read_rows(OLD_MONTH_RUN / "monthly_training_weights.csv")
    required_old_columns = {"training_loss", "zone", "month", "qweighted_oof_mae",
                            "difficulty_0_1", "raw_1_plus_0.5_score",
                            "final_training_weight", "used_in_final_training"}
    if not required_old_columns.issubset(old_rows[0]):
        raise RuntimeError(f"Old monthly weight file is missing columns: "
                           f"{required_old_columns - set(old_rows[0])}")
    old_weights: dict[tuple[str, str, str], float] = {}
    for row in old_rows:
        if row["used_in_final_training"].lower() == "true":
            old_weights[(row["training_loss"], row["zone"], row["month"])] = float(
                row["final_training_weight"])
    selected_info = json.loads((SELECTED_RUN / "run_info.json").read_text())
    configs = selected_info["configs"]
    if selected_info["input_profile_features"] != PROFILE_FEATURE_NAMES:
        raise RuntimeError("Selected profile feature set differs from the approved rerun plan")
    if selected_info["calendar_features"] != CALENDAR_FEATURE_NAMES:
        raise RuntimeError("Selected calendar feature set differs from the approved rerun plan")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "models").mkdir(exist_ok=True)
    scorer_progress = OUTPUT / "monthly_oof_difficulty.csv"
    scorer_rows = read_rows(scorer_progress) if scorer_progress.exists() else []
    scores = {objective: {zone: {} for zone in core.ZONES} for objective in OBJECTIVES}
    for row in scorer_rows:
        if row.get("scorer_config") != json.dumps(SCORER_CONFIG, sort_keys=True):
            raise RuntimeError("Saved scorer progress was produced with a different config")
        objective, zone, month = row["training_loss"], row["zone"], row["heldout_month"]
        if objective not in OBJECTIVES or zone not in core.ZONES or month not in months:
            raise RuntimeError(f"Unexpected scorer progress row: {row}")
        if month in scores[objective][zone]:
            raise RuntimeError(f"Duplicate scorer progress row for {objective}/{zone}/{month}")
        scores[objective][zone][month] = float(row["qweighted_mae"])

    weight_rows = []

    for objective_index, objective in enumerate(OBJECTIVES):
        for zone_index, zone in enumerate(core.ZONES):
            zone_ids = np.asarray([i for i, (_, z) in enumerate(keys) if z == zone])
            for month_index, held_month in enumerate(months):
                if held_month in scores[objective][zone]:
                    print(f"reuse score {objective} {zone} {held_month}: "
                          f"qMAE={scores[objective][zone][held_month]:.4f}", flush=True)
                    continue
                val_ids = np.asarray([i for i in zone_ids if month_of(keys[i][0]) == held_month])
                train_ids = np.asarray([i for i in zone_ids if month_of(keys[i][0]) != held_month])
                if set(month_of(keys[i][0]) for i in train_ids) != set(months) - {held_month}:
                    raise RuntimeError(f"LOMO fold for {zone}/{held_month} is not an 11-month fold")
                hmean = float(hourly[train_ids].mean())
                hscale = float(hourly[train_ids].std())
                profile = core.profile_features(hourly, hmean, hscale)
                seed = core.SEED + objective_index * 100000 + zone_index * 10000 + month_index
                started = time.perf_counter()
                predicted, met, best_epoch, parameter_count = fit_scorer(
                    objective, seed, SCORER_CONFIG, profile, target, weekday, hourly,
                    train_ids, val_ids, device)
                score = qweighted_mae(target[val_ids], predicted)
                scores[objective][zone][held_month] = score
                scorer_rows.append({
                    "training_loss": objective, "zone": zone,
                    "heldout_month": held_month, "train_month_count": 11,
                    "qweighted_mae": score, "plain_mae": met["mae"],
                    "q1_mae": met["q1_mae"], "q4_mae": met["q4_mae"],
                    "best_epoch": best_epoch, "parameter_count": parameter_count,
                    "scorer_config": json.dumps(SCORER_CONFIG, sort_keys=True),
                    "seconds": time.perf_counter() - started,
                })
                write_rows(scorer_progress, scorer_rows)
                print(f"score {objective} {zone} {held_month}: qMAE={score:.4f}, "
                      f"MAE={met['mae']:.4f}, epoch={best_epoch}", flush=True)

    new_weights: dict[tuple[str, str, str], float] = {}
    for objective in OBJECTIVES:
        for zone in core.ZONES:
            train_scores = np.asarray([scores[objective][zone][month] for month in train_months],
                                      dtype=np.float64)
            minimum, maximum = float(train_scores.min()), float(train_scores.max())
            span = maximum - minimum
            all_scores = np.asarray([scores[objective][zone][month] for month in months],
                                    dtype=np.float64)
            difficulty = (np.zeros_like(all_scores) if span < 1e-12
                          else (all_scores - minimum) / span)
            raw_weights = 1.0 + MONTH_STRENGTH * difficulty
            day_counts = np.asarray([
                sum(1 for i, (day, z) in enumerate(keys)
                    if z == zone and train_mask[i] and month_of(day) == month)
                for month in train_months
            ], dtype=np.float64)
            raw_train_weights = np.asarray([raw_weights[months.index(month)]
                                            for month in train_months])
            normalizer = float(np.average(raw_train_weights, weights=day_counts))
            normalized = raw_train_weights / normalizer
            if abs(float(np.average(normalized, weights=day_counts)) - 1.0) > 1e-12:
                raise AssertionError(f"Day-weighted training weights do not average to 1: {zone}")
            for month, weight in zip(train_months, normalized):
                new_weights[(objective, zone, month)] = float(weight)
            for month, score, diff, raw_weight in zip(months, all_scores, difficulty, raw_weights):
                is_train = month in train_months
                weight_rows.append({
                    "weight_scheme": "no_month_embedding_train_month_minmax" if is_train
                    else "diagnostic_only_train_month_minmax",
                    "training_loss": objective, "zone": zone, "month": month,
                    "qweighted_oof_mae": float(score), "difficulty_0_1": float(diff),
                    "raw_1_plus_0_5_score": float(raw_weight),
                    "final_training_weight": new_weights[(objective, zone, month)] if is_train else "",
                    "used_in_final_training": is_train,
                    "normalization_months": ";".join(train_months),
                    "profile_days": int(day_counts[train_months.index(month)]) if is_train else "",
                })
            for row in old_rows:
                if row["training_loss"] == objective and row["zone"] == zone:
                    weight_rows.append({
                        "weight_scheme": "old_month_embedding_all_12_minmax",
                        "training_loss": objective, "zone": zone, "month": row["month"],
                        "qweighted_oof_mae": row["qweighted_oof_mae"],
                        "difficulty_0_1": row["difficulty_0_1"],
                        "raw_1_plus_0_5_score": row["raw_1_plus_0.5_score"],
                        "final_training_weight": row["final_training_weight"],
                        "used_in_final_training": row["used_in_final_training"],
                        "normalization_months": "all_12_months",
                        "profile_days": "",
                    })

    # The final experiment is paired: identical model settings and seeds under
    # old versus corrected weights, for both loss choices.
    metric_rows, prediction_rows, comparison_rows = [], [], []
    for objective in OBJECTIVES:
        for zone_index, zone in enumerate(core.ZONES):
            zone_mask = np.asarray([z == zone for _, z in keys])
            train_ids = np.where(train_mask & zone_mask)[0]
            val_ids = np.where(val_mask & zone_mask)[0]
            hmean, hscale = float(hourly[train_ids].mean()), float(hourly[train_ids].std())
            all_features = core.profile_features(hourly, hmean, hscale)[:, :, PROFILE_CHANNELS]
            config = configs[zone]
            seed = core.SEED + zone_index * 10000 + 100 + SEED_OFFSETS[zone]
            zone_metric = {}
            for weighting in ("old_weights", "refit_weights"):
                sample_weights = np.ones(len(keys), dtype=np.float32)
                for idx in train_ids:
                    month = month_of(keys[idx][0])
                    sample_weights[idx] = (old_weights[(objective, zone, month)]
                                           if weighting == "old_weights"
                                           else new_weights[(objective, zone, month)])
                started = time.perf_counter()
                predicted, met, best_epoch, parameter_count, checkpoint = fit_selected(
                    objective, weighting, zone, seed, config, sample_weights,
                    all_features, hourly, target, weekday, month_ids, train_ids,
                    val_ids, device)
                zone_metric[weighting] = met
                metric_rows.append({
                    "objective": objective, "weighting": weighting, "zone": zone,
                    "train_months": ";".join(train_months),
                    "validation_period": f"{VAL_START}..{VAL_END}",
                    "mae": met["mae"], "qweighted_mae": met["qweighted_mae"],
                    "q1_mae": met["q1_mae"], "q2_mae": met["q2_mae"],
                    "q3_mae": met["q3_mae"], "q4_mae": met["q4_mae"],
                    "best_epoch": best_epoch, "parameter_count": parameter_count,
                    "seed": seed, "config": json.dumps(config, sort_keys=True),
                    "seconds": time.perf_counter() - started,
                })
                for j, idx in enumerate(val_ids):
                    for h in range(24):
                        for q in range(4):
                            prediction_rows.append([
                                objective, weighting, keys[idx][0], zone, h + 1, q + 1,
                                float(target[idx, h, q]), float(predicted[j, h, q])])
                if weighting == "refit_weights":
                    checkpoint.update({
                        "month_weights": {m: new_weights[(objective, zone, m)]
                                          for m in train_months},
                        "hourly_mean_train": hmean,
                        "hourly_std_train": hscale,
                    })
                    model_path = OUTPUT / "models" / f"{objective}_{zone}_no_month_embedding.pt"
                    torch.save(checkpoint, model_path)
                print(f"final {objective} {zone} {weighting}: MAE={met['mae']:.5f}, "
                      f"qMAE={met['qweighted_mae']:.5f}, epoch={best_epoch}", flush=True)

            old_mae, new_mae = zone_metric["old_weights"]["mae"], zone_metric["refit_weights"]["mae"]
            comparison_rows.append({
                "objective": objective, "zone": zone,
                "old_weight_mae": old_mae, "refit_weight_mae": new_mae,
                "mae_change_new_minus_old": new_mae - old_mae,
                "mae_change_percent": 100.0 * (new_mae - old_mae) / old_mae,
            })

    legacy_metrics = {r["zone"]: float(r["mae"])
                      for r in read_rows(SELECTED_RUN / "validation_metrics.csv")}
    for row in comparison_rows:
        row["existing_selected_mae_baseline"] = legacy_metrics[row["zone"]] \
            if row["objective"] == "mae" else ""
        row["refit_mae_change_vs_existing_selected"] = (
            float(row["refit_weight_mae"]) - legacy_metrics[row["zone"]]
            if row["objective"] == "mae" else "")
    for objective in OBJECTIVES:
        subset = [r for r in comparison_rows if r["objective"] == objective]
        old_mean = float(np.mean([r["old_weight_mae"] for r in subset]))
        new_mean = float(np.mean([r["refit_weight_mae"] for r in subset]))
        comparison_rows.append({
            "objective": objective, "zone": "MEAN_ZONES",
            "old_weight_mae": old_mean, "refit_weight_mae": new_mean,
            "mae_change_new_minus_old": new_mean - old_mean,
            "mae_change_percent": 100.0 * (new_mean - old_mean) / old_mean,
            "existing_selected_mae_baseline": float(np.mean(list(legacy_metrics.values())))
            if objective == "mae" else "",
            "refit_mae_change_vs_existing_selected": new_mean - float(
                np.mean(list(legacy_metrics.values()))) if objective == "mae" else "",
        })

    write_rows(OUTPUT / "monthly_oof_difficulty.csv", scorer_rows)
    write_rows(OUTPUT / "monthly_training_weights.csv", weight_rows)
    write_rows(OUTPUT / "aug_sep_metrics.csv", metric_rows)
    write_rows(OUTPUT / "comparison_summary.csv", comparison_rows)
    with (OUTPUT / "aug_sep_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["objective", "weighting", "delivery_date", "market_id",
                         "hour_index", "quarter_index", "actual_eur_mwh", "predicted_eur_mwh"])
        writer.writerows(prediction_rows)

    run_info = {
        "device": str(device),
        "source": str(SOURCE.relative_to(ROOT)),
        "all_12_months": months,
        "final_training_period": ["2025-10-01", "2026-07-31"],
        "final_validation_period": [VAL_START, VAL_END],
        "scorer_experiment": "per-zone leave-one-month-out CNN; train on 11 months, score held-out month",
        "scorer_architecture": SCORER_CONFIG,
        "scorer_calendar_features": ["hour_embedding_6", "weekday_embedding_4"],
        "scorer_month_embedding": False,
        "scorer_objectives": list(OBJECTIVES),
        "difficulty_metric": "Q-weighted held-out MAE with quarter weights 1.5,1,1,1.5",
        "weight_formula": "within-zone min-max difficulty using October-July scores only; raw=1+0.5*difficulty; normalize to day-weighted mean 1 over October-July",
        "validation_month_scores_used_for_training_weights": False,
        "final_architecture": "one selected reduced-feature CNN per zone",
        "final_profile_features": PROFILE_FEATURE_NAMES,
        "final_calendar_features": CALENDAR_FEATURE_NAMES,
        "final_month_embedding": False,
        "final_configs": configs,
        "final_seed_offsets": SEED_OFFSETS,
        "final_objectives": list(OBJECTIVES),
        "final_checkpoint_selection_metric": "plain unweighted August-September MAE",
        "final_quarter_weights": list(core.Q_WEIGHT),
        "paired_weight_comparison": "same final model config and seed per objective/zone under prior and refit weights",
        "historical_old_weight_source": str((OLD_MONTH_RUN / "monthly_training_weights.csv").relative_to(ROOT)),
        "existing_selected_baseline_source": str((SELECTED_RUN / "validation_metrics.csv").relative_to(ROOT)),
    }
    (OUTPUT / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    for objective in OBJECTIVES:
        row = next(r for r in comparison_rows if r["objective"] == objective
                   and r["zone"] == "MEAN_ZONES")
        print(f"comparison {objective}: old={float(row['old_weight_mae']):.5f}, "
              f"refit={float(row['refit_weight_mae']):.5f}, "
              f"delta={float(row['mae_change_new_minus_old']):+.5f}", flush=True)
    print(f"wrote experiment artifacts to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
