"""Reproduce the frozen six-network recipe directly from the official CSV.

Adapted from campaigns/submission_champion_20261017/refit.py. Mode 'trial' is
an execution check with four fit days, two calibration days and one epoch per
component; mode 'final' keeps the original 348/15 split and all fixed budgets.
"""
import argparse
import json
import os
from pathlib import Path
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import torch

import feature_pipeline as p
import quantile_models as q
from pipeline_utils import (ROOT, check_causality, environment, feature_data,
                            read_history, seed_everything, sha256, write_json)
from predict_prices import predict_arrays
from price_data import baseline


def train(mode, device, data_path, output):
    started = time.perf_counter()
    recipe_path = ROOT / 'config/final_recipe.json'
    provenance = json.loads((ROOT / 'provenance/source_map.json').read_text())
    if sha256(recipe_path) != provenance['copied_files']['config/final_recipe.json']:
        raise ValueError('The frozen recipe has changed')
    recipe = json.loads(recipe_path.read_text())
    if sha256(data_path) != recipe['official_sha256']:
        raise ValueError('Training requires the unchanged official training CSV (SHA-256 mismatch)')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a suitable PyTorch build or use --device cpu.')
    output = Path(output)
    for name in ['model.pt', 'training_report.json', 'calibration_replay.npz']:
        if (output / name).exists():
            raise FileExistsError(f'{output / name}: use a fresh output directory')
    dates, y = read_history(data_path)
    dates = dates.astype(object)
    if len(dates) != 363 or dates[0] != '2025-10-01' or dates[-1] != '2026-09-30':
        raise ValueError('Unexpected official date coverage')
    if mode == 'trial':
        dates, y = dates[:6], y[:6]
        mask = np.arange(len(dates)) < 4
    else:
        mask = (dates >= recipe['training_start']) & (dates <= recipe['training_end'])
        calibration_mask = (dates >= recipe['calibration_start']) & (dates <= recipe['calibration_end'])
        if int(mask.sum()) != recipe['training_days'] or int(calibration_mask.sum()) != recipe['calibration_days'] or not (mask | calibration_mask).all():
            raise ValueError('The frozen training/calibration split does not match the input')
    nt, nc = int(mask.sum()), int((~mask).sum())
    hourly = y.mean(-1)
    anchor = baseline(hourly)
    probes = check_causality(dates, hourly, dates, y, [0, nt - 1, nt, len(dates) - 1])
    data = feature_data(dates, hourly, dates, y, fit_mask=mask)
    weights, details = p.weights_for(dates, y, anchor, mask, 'ratio')
    seed_everything(17)
    target = torch.tensor((y[mask] - anchor[mask]) / 20, dtype=torch.float32, device=device)
    tau = torch.tensor(q.TAUS, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    models = []
    cal_q = np.zeros((*y[~mask].shape, 11), dtype='float64')
    components = []
    for leaf in recipe['leaves']:
        cfg = leaf['config']
        seed_everything(cfg['seed'])
        net = q.Quantiles(cfg, data)
        net.initialize((y[mask] - anchor[mask]) / 20)
        net.to(device)
        parameters = sum(value.numel() for value in net.parameters())
        if parameters != leaf['parameters']:
            raise ValueError(f'Architecture changed for {cfg["family"]}')
        x = torch.tensor(data[net.feature][mask], device=device)
        curve = torch.tensor(data['curve'][mask], device=device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=.01)
        epochs = 1 if mode == 'trial' else leaf['best_epoch']
        batch_size = 2 if mode == 'trial' else 32
        for epoch in range(1, epochs + 1):
            net.train()
            for ids in torch.randperm(nt, device=device).split(batch_size):
                predicted = net(x[ids], curve[ids])
                error = target[ids, ..., None] - predicted
                mae = error[..., 5].abs().mean((1, 2, 3))
                pinball = torch.maximum(tau * error, (tau - 1) * error).mean((1, 2, 3, 4))
                loss = ((mae + cfg['quantile_weight'] * pinball) * weight_tensor[ids]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
            if epoch % 20 == 0:
                print('FIT', cfg['family'], cfg['seed'], epoch, '/', epochs, flush=True)
        net.eval()
        with torch.no_grad():
            raw = net(torch.tensor(data[net.feature][~mask], device=device),
                      torch.tensor(data['curve'][~mask], device=device)).cpu().numpy()
        cal_q += leaf['weight'] * q.physical(raw, anchor[~mask])
        models.append(dict(config=cfg, weight=leaf['weight'], epochs=epochs,
                           state={key: value.cpu().clone() for key, value in net.state_dict().items()}))
        components.append(dict(family=cfg['family'], seed=cfg['seed'], epochs=epochs,
                               parameters=parameters, weight=leaf['weight'], last_loss=float(loss.detach().cpu())))
        print('COMPONENT COMPLETE', cfg['family'], cfg['seed'], 'epochs', epochs, flush=True)
    bands = q.intervals(cal_q, y[~mask], cal_q)
    calibration = {str(level): {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in item.items()
                              if key not in ['lower', 'upper', 'cal_lower', 'cal_upper']}
                   for level, item in bands.items()}
    metadata = dict(artifact_kind='isolated_trial' if mode == 'trial' else 'jury_full_retrain',
                    mode=mode, device=device, training_dates=dates[mask].tolist(),
                    calibration_dates=dates[~mask].tolist(), official_sha256=sha256(data_path),
                    fixed_budget_no_calibration_stopping=True,
                    original_frozen_weights_supplied=False,
                    batch_size=2 if mode == 'trial' else 32,
                    epoch_override=1 if mode == 'trial' else None)
    bundle = dict(schema_version=1, recipe=recipe, models=models, calibration=calibration,
                  training_days=nt, calibration_days=nc, training_month_baselines=details,
                  run_metadata=metadata)
    replay = predict_arrays(bundle, dates[~mask], hourly[~mask], dates, y)
    replay_error = float(np.abs(replay - cal_q).max())
    if replay_error > .0001:
        raise ValueError(f'CPU state replay mismatch: {replay_error}')
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output / 'model.pt')
    np.savez(output / 'calibration_replay.npz', dates=dates[~mask].astype('U10'),
             hourly=hourly[~mask], observed=y[~mask], quantiles=cal_q)
    write_json(output / 'calibration.json', calibration)
    write_json(output / 'environment.json', environment())
    report = dict(completed=True, **metadata, training_days=nt, calibration_days=nc,
                  components=components, source_commit=provenance['source_commit'],
                  feature_shapes={key: list(value.shape) for key, value in data.items()},
                  causal_probes=probes, cpu_replay_max_error=replay_error,
                  max_hourly_mean_error=float(np.abs(cal_q[..., 5].mean(-1) - hourly[~mask]).max()),
                  seconds=time.perf_counter() - started, model_sha256=sha256(output / 'model.pt'),
                  environment=environment(), accuracy_evaluated=False)
    write_json(output / 'training_report.json', report)
    write_json(output / 'SHA256SUMS.json', {path.name: sha256(path) for path in sorted(output.iterdir())
                                         if path.is_file() and path.name != 'SHA256SUMS.json'})
    print('TRAINING COMPLETE', mode, 'checkpoint:', output / 'model.pt', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['trial', 'final'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/train_prices_15min.csv')
    parser.add_argument('--output', type=Path, required=True, help='Fresh checkpoint directory')
    args = parser.parse_args()
    train(args.mode, args.device, args.data, args.output)


if __name__ == '__main__':
    main()
