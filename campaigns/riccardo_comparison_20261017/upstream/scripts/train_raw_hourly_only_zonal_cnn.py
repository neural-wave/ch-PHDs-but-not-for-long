#!/usr/bin/env python3
"""Train selected per-zone CNNs using only the raw 24 hourly prices."""
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
SELECTED_RUN = ROOT / "experiments" / "selected_reduced_features_per_zone"
REFIT_RUN = ROOT / "experiments" / "month_difficulty_no_month_embedding_aug_sep"
COMPOSITION_RUN = ROOT / "experiments" / "selected_feature_composition_ablation"
OUTPUT = ROOT / "experiments" / "raw_hourly_only_zonal_cnn"

VAL_START, VAL_END = "2026-08-01", "2026-09-30"
MAX_EPOCHS, PATIENCE = 150, 16
SEED_OFFSETS = {"ZONE_0": 4, "ZONE_1": 7, "ZONE_2": 6, "ZONE_3": 8}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(predicted - actual)
    qweights = np.asarray(core.Q_WEIGHT, dtype=np.float64).reshape(1, 1, 4)
    return {
        "mae": float(error.mean()),
        "qweighted_mae": float((error * qweights).mean() / qweights.mean()),
        **{f"q{q + 1}_mae": float(error[:, :, q].mean()) for q in range(4)},
    }


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
    train_mask = np.asarray([day < VAL_START for day, _ in keys])
    val_mask = np.asarray([VAL_START <= day <= VAL_END for day, _ in keys])
    train_months = sorted({day[:7] for (day, _), keep in zip(keys, train_mask) if keep})

    selected_info = json.loads((SELECTED_RUN / "run_info.json").read_text())
    configs = selected_info["configs"]
    weight_rows = read_rows(REFIT_RUN / "monthly_training_weights.csv")
    month_weights = {
        (row["training_loss"], row["zone"], row["month"]): float(row["final_training_weight"])
        for row in weight_rows
        if row["weight_scheme"] == "no_month_embedding_train_month_minmax"
        and row["training_loss"] == "mae"
        and row["used_in_final_training"].lower() == "true"
    }
    if len(month_weights) != 40:
        raise RuntimeError(f"Expected 40 revised MAE month weights, found {len(month_weights)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "models").mkdir(exist_ok=True)
    qweights = torch.tensor(core.Q_WEIGHT, dtype=torch.float32, device=device).view(1, 1, 4)
    metric_rows, prediction_rows, model_info = [], [], {}

    for zone_index, zone in enumerate(core.ZONES):
        zone_ids = np.asarray([i for i, (_, z) in enumerate(keys) if z == zone])
        train_ids = zone_ids[train_mask[zone_ids]]
        val_ids = zone_ids[val_mask[zone_ids]]
        if len(train_ids) != 302 or len(val_ids) != 61:
            raise RuntimeError(f"Unexpected split sizes for {zone}: {len(train_ids)}/{len(val_ids)}")

        # The sole input channel is the original hourly price in EUR/MWh.
        # No centering, scaling, derivative channels, or calendar embeddings.
        raw_hourly_input = hourly[:, :, None].astype(np.float32)
        target_scale = float(np.mean(np.abs(target[train_ids] - hourly[train_ids, :, None])))
        sample_weight = np.ones(len(keys), dtype=np.float32)
        for idx in train_ids:
            sample_weight[idx] = month_weights[("mae", zone, keys[idx][0][:7])]

        config = configs[zone]
        seed = core.SEED + zone_index * 10000 + 100 + SEED_OFFSETS[zone]
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        model = ZonalAblationCNN(config, profile_count=1, calendar_features=[]).to(device)
        if any("hour" in name.lower() or "weekday" in name.lower()
               or "month" in name.lower() for name, _ in model.named_parameters()):
            raise AssertionError("Raw-hourly model unexpectedly has calendar embeddings")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["wd"])

        x = torch.as_tensor(raw_hourly_input[train_ids], dtype=torch.float32, device=device)
        y = torch.as_tensor(target[train_ids], dtype=torch.float32, device=device)
        mw = torch.as_tensor(sample_weight[train_ids], dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(x, y, mw), batch_size=config["batch"], shuffle=True)
        vx = torch.as_tensor(raw_hourly_input[val_ids], dtype=torch.float32, device=device)
        best_score, best_state, best_epoch, waiting = float("inf"), None, 0, 0
        started = time.perf_counter()

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for bx, by, bmonth in loader:
                optimizer.zero_grad(set_to_none=True)
                raw = model(bx, None, None) * target_scale
                predicted = core.center(raw, by.mean(dim=-1))
                loss = ((predicted - by).abs() * qweights * bmonth[:, None, None]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                raw = model(vx, None, None).cpu().numpy().astype(np.float64) * target_scale
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
            raise RuntimeError(f"No best checkpoint found for {zone}")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            raw = model(vx, None, None).cpu().numpy().astype(np.float64) * target_scale
        predicted = hourly[val_ids, :, None] + raw - raw.mean(axis=-1, keepdims=True)
        met = metrics(target[val_ids], predicted)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        checkpoint = {
            "state_dict": best_state,
            "zone": zone,
            "config": config,
            "input": "raw hourly prices, EUR/MWh, no input normalization",
            "input_feature_names": ["raw_hourly_price"],
            "calendar_features": [],
            "hour_embedding": False,
            "weekday_embedding": False,
            "month_embedding": False,
            "quarter_weights": list(core.Q_WEIGHT),
            "month_weight_strength": 0.5,
            "month_weights": {m: month_weights[("mae", zone, m)] for m in train_months},
            "target_residual_scale": target_scale,
            "seed": seed,
            "best_epoch": best_epoch,
            "parameter_count": parameter_count,
        }
        model_path = OUTPUT / "models" / f"{zone}_raw_hourly_cnn.pt"
        torch.save(checkpoint, model_path)
        metric_rows.append({
            "zone": zone, "seed": seed, "config": json.dumps(config, sort_keys=True),
            "input": "24 raw hourly prices; no input normalization; no calendar embeddings",
            "train_months": ";".join(train_months),
            "validation_period": f"{VAL_START}..{VAL_END}",
            "mae": met["mae"], "qweighted_mae": met["qweighted_mae"],
            "q1_mae": met["q1_mae"], "q2_mae": met["q2_mae"],
            "q3_mae": met["q3_mae"], "q4_mae": met["q4_mae"],
            "best_epoch": best_epoch, "parameter_count": parameter_count,
            "seconds": time.perf_counter() - started,
        })
        for j, idx in enumerate(val_ids):
            for h in range(24):
                for q in range(4):
                    prediction_rows.append([keys[idx][0], zone, h + 1, q + 1,
                                            float(target[idx, h, q]), float(predicted[j, h, q])])
        model_info[zone] = {"checkpoint": str(model_path.relative_to(ROOT)),
                            "parameter_count": parameter_count, "best_epoch": best_epoch,
                            "validation_metrics": met}
        print(f"raw-hourly {zone}: MAE={met['mae']:.5f}, qMAE={met['qweighted_mae']:.5f}, "
              f"epoch={best_epoch}", flush=True)

    composition = {r["zone"]: float(r["mae_mean"]) for r in read_rows(
        COMPOSITION_RUN / "ablation_summary.csv")
        if r["variant"] == "profile_only" and r["zone"] in core.ZONES}
    current_full = {r["zone"]: float(r["mae"]) for r in read_rows(
        SELECTED_RUN / "validation_metrics.csv")}
    current_refit = {r["zone"]: float(r["mae"]) for r in read_rows(
        REFIT_RUN / "aug_sep_metrics.csv")
        if r["objective"] == "mae" and r["weighting"] == "refit_weights"}
    comparison_rows = []
    for row in metric_rows:
        zone = row["zone"]
        comparison_rows.append({
            "zone": zone, "raw_24_hour_input_mae": row["mae"],
            "four_profile_channels_no_calendar_mae": composition[zone],
            "current_selected_full_mae_existing": current_full[zone],
            "current_selected_full_mae_refit_weights": current_refit[zone],
            "raw_vs_four_profile_delta": row["mae"] - composition[zone],
        })
    comparison_rows.append({
        "zone": "MEAN_ZONES",
        "raw_24_hour_input_mae": float(np.mean([r["mae"] for r in metric_rows])),
        "four_profile_channels_no_calendar_mae": float(np.mean(list(composition.values()))),
        "current_selected_full_mae_existing": float(np.mean(list(current_full.values()))),
        "current_selected_full_mae_refit_weights": float(np.mean(list(current_refit.values()))),
        "raw_vs_four_profile_delta": float(np.mean([r["mae"] for r in metric_rows]))
        - float(np.mean(list(composition.values()))),
    })

    write_rows(OUTPUT / "validation_metrics.csv", metric_rows)
    write_rows(OUTPUT / "comparison_summary.csv", comparison_rows)
    with (OUTPUT / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["delivery_date", "market_id", "hour_index", "quarter_index",
                         "actual_eur_mwh", "predicted_eur_mwh"])
        writer.writerows(prediction_rows)
    run_info = {
        "device": str(device),
        "architecture": "one CNN per zone; same selected width/depth/dropout/optimizer config",
        "input": "24 raw hourly prices in EUR/MWh, supplied as one unnormalized channel",
        "input_normalization": False,
        "derived_profile_features": False,
        "calendar_embeddings": False,
        "training_period": ["2025-10-01", "2026-07-31"],
        "validation_period": [VAL_START, VAL_END],
        "loss": "Q1/Q4-weighted MAE; quarter weights 1.5,1,1,1.5; revised month weights with strength 0.5",
        "month_weights_source": str((REFIT_RUN / "monthly_training_weights.csv").relative_to(ROOT)),
        "validation_checkpoint_selection": "plain unweighted MAE",
        "configs": configs,
        "models": model_info,
        "note": "The model still scales its predicted residual by a train-only residual target scale; the 24 hourly input values themselves are passed unchanged.",
    }
    (OUTPUT / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(f"wrote raw-hourly experiment to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
