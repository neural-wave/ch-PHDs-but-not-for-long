"""Re-evaluate the pinned riccardo CNNs under VALIDATION_PROTOCOL.md.

All fitting runs on the authorized research GPU host. The upstream code is
vendored for reproducibility; no external/October files enter this campaign.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'campaigns/riccardo_comparison_20261017'
sys.path.insert(0, str(CAMPAIGN / 'upstream/scripts'))
SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
METHODS = ['riccardo_reduced', 'riccardo_raw_hourly']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    import numpy as np
    import validate_aug_sep as core
    assert hashlib.sha256((ROOT / 'train_prices_15min.csv').read_bytes()).hexdigest() == SHA
    keys, h, y = core.read_profiles(ROOT / 'train_prices_15min.csv')
    assert len(keys) == 363 * 4 and len(set(keys)) == len(keys)
    wd, mo = core.calendar(keys)
    month = np.array([d[:7] for d, _ in keys])
    day = np.array([int(d[8:10]) for d, _ in keys])
    zones = np.array([z for _, z in keys])
    return keys, h, y, wd, mo, month, day, zones


def fold_specs():
    months = ['2025-10', '2025-11', '2025-12'] + [f'2026-{m:02d}' for m in range(1, 10)]
    return [dict(months=i + 1, train_months=months[:i + 1], validation=months[i + 1],
                 test=months[i + 2]) for i in range(2, 10)]


def fit_final(method, config, seed, weights, x, h, y, wd, mo, train, select, device):
    """Upstream optimization/early stopping, with outer selection isolated."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import validate_aug_sep as core
    import refit_month_difficulty_no_month_embedding as upstream
    from feature_ablation_per_zone import ZonalAblationCNN
    upstream.seed_everything(seed)
    calendar = upstream.CALENDAR_FEATURE_NAMES if method == 'riccardo_reduced' else []
    model = ZonalAblationCNN(config, x.shape[-1], calendar).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    scale = float(np.mean(np.abs(y[train] - h[train, :, None])))
    tensor = lambda v, dtype=torch.float32: torch.as_tensor(v, dtype=dtype, device=device)
    loader = DataLoader(TensorDataset(tensor(x[train]), tensor(y[train]),
                        tensor(wd[train], torch.long), tensor(mo[train], torch.long),
                        tensor(weights[train])), batch_size=config['batch'], shuffle=True)
    vx, vw, vm = tensor(x[select]), tensor(wd[select], torch.long), tensor(mo[select], torch.long)
    qw = tensor(core.Q_WEIGHT).view(1, 1, 4)
    best, state, best_epoch, waiting, history = float('inf'), None, 0, 0, []
    for epoch in range(1, upstream.MAX_EPOCHS + 1):
        model.train()
        for bx, by, bw, bm, mw in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = core.center(model(bx, bw, bm) * scale, by.mean(-1))
            loss = ((pred - by).abs() * qw * mw[:, None, None]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            raw = model(vx, vw, vm).cpu().numpy().astype('float64') * scale
        pred = h[select, :, None] + raw - raw.mean(-1, keepdims=True)
        score = float(np.abs(pred - y[select]).mean())
        history.append(dict(epoch=epoch, selection_mae=score))
        if score < best - 1e-6:
            best, best_epoch, waiting = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= upstream.PATIENCE:
                break
    assert state is not None
    return dict(state_dict=state, config=config, method=method, calendar=calendar,
                profile_count=x.shape[-1], target_residual_scale=scale, seed=seed,
                best_epoch=best_epoch, selection_mae=best, history=history,
                parameter_count=sum(p.numel() for p in model.parameters()))


def predict(checkpoint, x, h, wd, mo, ids, device):
    import torch
    from feature_ablation_per_zone import ZonalAblationCNN
    model = ZonalAblationCNN(checkpoint['config'], checkpoint['profile_count'], checkpoint['calendar']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        raw = model(torch.as_tensor(x[ids], dtype=torch.float32, device=device),
                    torch.as_tensor(wd[ids], dtype=torch.long, device=device),
                    torch.as_tensor(mo[ids], dtype=torch.long, device=device)).cpu().numpy().astype('float64')
    raw *= checkpoint['target_residual_scale']
    return h[ids, :, None] + raw - raw.mean(-1, keepdims=True)


def task(spec, zone_index, out, smoke):
    import numpy as np
    import torch
    import validate_aug_sep as core
    import refit_month_difficulty_no_month_embedding as upstream
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), 'Research computation requires the authorized CUDA server'
    device = torch.device('cuda')
    if smoke:
        upstream.MAX_EPOCHS = 2
    keys, h, y, wd, mo, month, day, zones = load()
    zone = core.ZONES[zone_index]
    train = np.flatnonzero((zones == zone) & np.isin(month, spec['train_months']))
    select = np.flatnonzero((zones == zone) & (month == spec['validation']) & (day <= 15))
    cal = np.flatnonzero((zones == zone) & (month == spec['validation']) & (day > 15))
    test = np.flatnonzero((zones == zone) & (month == spec['test']))
    assert keys[train[-1]][0] < keys[select[0]][0] < keys[cal[0]][0] < keys[test[0]][0]
    folder = Path(out) / f"{spec['test']}/{zone}"
    folder.mkdir(parents=True, exist_ok=True)
    scores = []
    for month_index, held in enumerate(spec['train_months']):
        inner_train, held_ids = train[month[train] != held], train[month[train] == held]
        assert not set(inner_train) & set(held_ids)
        assert set(inner_train) | set(held_ids) == set(train)
        mean, std = float(h[inner_train].mean()), float(h[inner_train].std())
        x = core.profile_features(h, mean, std)
        seed = core.SEED + zone_index * 10000 + month_index
        pp, met, epoch, params = upstream.fit_scorer('mae', seed, upstream.SCORER_CONFIG,
            x, y, wd, h, inner_train, held_ids, device)
        score = upstream.qweighted_mae(y[held_ids], pp)
        scores.append(dict(held_month=held, qweighted_mae=score, best_epoch=epoch,
                           train_dates=[keys[i][0] for i in inner_train], held_dates=[keys[i][0] for i in held_ids],
                           hourly_mean_train=mean, hourly_std_train=std))
    values = np.array([r['qweighted_mae'] for r in scores])
    span = float(values.max() - values.min())
    difficulty = np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span
    raw = 1 + 0.5 * difficulty
    counts = np.array([sum(month[train] == m) for m in spec['train_months']])
    final = raw / np.average(raw, weights=counts)
    weights_by_month = dict(zip(spec['train_months'], final.tolist()))
    sample_weight = np.ones(len(keys), dtype='float32')
    sample_weight[train] = [weights_by_month[m] for m in month[train]]
    mean, std = float(h[train].mean()), float(h[train].std())
    configs = json.loads((CAMPAIGN / 'upstream/experiments/selected_reduced_features_per_zone/run_info.json').read_text())['configs']
    report = dict(spec=spec, zone=zone, train_dates=[keys[i][0] for i in train],
                  selection_dates=[keys[i][0] for i in select], calibration_dates=[keys[i][0] for i in cal],
                  test_dates=[keys[i][0] for i in test], scorer=scores, month_weights=weights_by_month,
                  hourly_mean_train=mean, hourly_std_train=std, methods={}, smoke=smoke)
    dump(folder / 'weight_receipt.json', report)
    seed = core.SEED + zone_index * 10000 + 100 + upstream.SEED_OFFSETS[zone]
    for method in METHODS:
        x = (core.profile_features(h, mean, std)[:, :, upstream.PROFILE_CHANNELS]
             if method == 'riccardo_reduced' else h[:, :, None].astype('float32'))
        checkpoint = fit_final(method, configs[zone], seed, sample_weight, x, h, y, wd, mo, train, select, device)
        if smoke:
            altered = y.copy()
            altered[np.r_[cal, test]] += np.array([-3000, -1000, 1000, 3000])
            repeated = fit_final(method, configs[zone], seed, sample_weight, x, h, altered, wd, mo, train, select, device)
            assert checkpoint['best_epoch'] == repeated['best_epoch']
            assert max(float((v-repeated['state_dict'][k]).abs().max()) for k,v in checkpoint['state_dict'].items()) < 1e-6
            if method == 'riccardo_reduced':
                _, _, epoch, _, original = upstream.fit_selected('mae', 'train_only', zone, seed,
                    configs[zone], sample_weight, x, h, y, wd, mo, train, select, device)
                assert epoch == checkpoint['best_epoch']
                assert max(float((v-original['state_dict'][k]).abs().max()) for k,v in checkpoint['state_dict'].items()) < 1e-6
        checkpoint.update(hourly_mean_train=mean, hourly_std_train=std, spec=spec, zone=zone,
                          train_dates=report['train_dates'], month_weights=weights_by_month)
        # Persist the selected state before any test scoring.
        torch.save(checkpoint, folder / (method + '.pt'))
        replay = torch.load(folder / (method + '.pt'), map_location='cpu', weights_only=False)
        ids = np.r_[select, cal, test]
        pp = predict(replay, x, h, wd, mo, ids, device)
        assert np.isfinite(pp).all() and np.abs(pp.mean(-1) - h[ids]).max() < 1e-8
        selection_mae = float(np.abs(pp[:len(select)] - y[select]).mean())
        assert abs(selection_mae - replay['selection_mae']) < 1e-7
        np.savez_compressed(folder / (method + '.npz'), ids=ids, dates=np.array([keys[i][0] for i in ids]),
            actual=y[ids], predicted=pp, selection_count=len(select), calibration_count=len(cal))
        err = np.abs(pp[-len(test):] - y[test])
        report['methods'][method] = dict(test_mae=float(err.mean()), selection_mae=selection_mae,
            best_epoch=replay['best_epoch'], parameter_count=replay['parameter_count'],
            daily_mae=err.mean((1, 2)).tolist())
    dump(folder / 'metrics.json', report)
    return str(folder / 'metrics.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = ROOT / ('results/riccardo_comparison_20261017' + ('_smoke' if args.smoke else ''))
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'metrics.json').exists(), 'Do not overwrite a completed campaign'
    receipt = json.loads((CAMPAIGN / 'upstream_receipt.json').read_text())
    for path, expected in receipt['files'].items():
        assert hashlib.sha256((CAMPAIGN / 'upstream' / path).read_bytes()).hexdigest() == expected
    specs = fold_specs()
    if args.smoke:
        specs = [specs[0], specs[-1]]
    dump(out / 'manifest.json', dict(commit=receipt['commit'], source_sha256=SHA, methods=METHODS,
         folds=specs, smoke=args.smoke, workers=args.workers, max_epochs=2 if args.smoke else 150,
         patience=16, config_policy='Freeze upstream selected per-zone configs; new first-15-day epoch selection',
         weighting='Fresh MAE LOMO scoring entirely within each outer training block; no cached upstream scores',
         calibration='Days 16-end unused for fitting or selection; no intervals requested',
         source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                        ['compare_riccardo.py','audit_riccardo_comparison.py','VALIDATION_PROTOCOL.md']}))
    started, reports = time.time(), []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(task, spec, z, str(out), args.smoke) for spec in specs for z in range(4)]
        for future in as_completed(jobs):
            path = future.result()
            reports.append(json.loads(Path(path).read_text()))
            dump(out / 'progress.json', dict(completed_zone_folds=len(reports), total_zone_folds=len(jobs),
                 elapsed_seconds=time.time()-started, last=path))
            print('DONE', len(reports), '/', len(jobs), path, flush=True)
    reports.sort(key=lambda r:(r['spec']['test'], r['zone']))
    dump(out / 'metrics.json', dict(reports=reports, elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    main()
