"""Fixed physical scaling and conservative blends, with growing training history."""
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'campaigns/monthly_quantiles_20261017'))
import trainq as q
import numpy as np
import pandas as pd
import torch

p = q.p
ROOT = PROJECT / 'results/minimize_expanding_20261017'
OLD = PROJECT / 'results/monthly_quantiles_20261017'
q.ROOT = p.ROOT = ROOT
TAUS, INTERVALS, Quantiles, physical = q.TAUS, q.INTERVALS, q.Quantiles, q.physical


def configs():
    return [dict(**c, preprocessing='fixed_physical') for c in q.configs()]


def prepare():
    p.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((OLD / 'manifest.json').read_text())
    assert json.loads((OLD / 'audit.json').read_text())['passed']
    for path, sha in prior['code_hashes'].items():
        assert hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() == sha, path
    assert hashlib.sha256((PROJECT / 'train_prices_15min.csv').read_bytes()).hexdigest() == p.SHA
    raw = np.load(OLD / 'raw.npz')
    assert hashlib.sha256((OLD / 'raw.npz').read_bytes()).hexdigest() == prior['feature_source_sha256']
    dates = raw['dates'].astype(object)  # pandas requires Python str in causal feature probes.
    # Repeat the causal probes for the exact reused features and current source.
    for i in [0, 91, 150, 302, 333]:
        y = raw['y']; changed = y.copy(); changed[i:] += np.arange(4) * 123456
        probe = p.features(dates[i:i+1], raw['h'][i:i+1], dates, changed)
        trunc = p.features(dates[i:i+1], raw['h'][i:i+1], dates[:i], y[:i])
        for key in probe:
            np.testing.assert_allclose(probe[key], raw[key][i:i+1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(probe[key], trunc[key], atol=1e-9, rtol=0)
    paths = list(prior['code_hashes']) + [str((HERE / f).relative_to(PROJECT))
            for f in ['train.py', 'audit.py', 'audit_selection.py', 'run.py', 'PROTOCOL.md']]
    manifest = dict(source_sha256=p.SHA, configs=configs(), folds=prior['folds'],
                    taus=TAUS.tolist(), feature_source_sha256=prior['feature_source_sha256'],
                    code_hashes={f: hashlib.sha256((PROJECT / f).read_bytes()).hexdigest() for f in paths},
                    original_manifest_sha256=hashlib.sha256((OLD / 'manifest.json').read_bytes()).hexdigest(),
                    causal_perturbation_and_truncation=True, expanding_training=True,
                    selection='preceding-month days1-15; calibration16-end; full next-month test')
    if (ROOT / 'manifest.json').exists():
        assert json.loads((ROOT / 'manifest.json').read_text()) == manifest, 'Use a new campaign for changed source'
    shutil.copy2(OLD / 'raw.npz', ROOT / 'raw.npz')
    p.write(ROOT / 'manifest.json', manifest)
    print('PREPARED', len(configs()), 'new fits per expanding fold', flush=True)


def folder_data(fold, smoke=False):
    folder = p.fold_data(fold, smoke)
    data = dict(np.load(folder / 'data.npz'))
    raw = np.load(ROOT / 'raw.npz'); nt = int(data['nt'])
    ids = np.searchsorted(raw['dates'], data['dates'])
    for feature in ['rich', 'compact', 'cnn', 'curve']:
        values = np.clip(raw[feature][ids], -15, 15).astype('float32')
        data[feature + '_mu'] = np.zeros_like(data[feature + '_mu'])
        data[feature + '_sd'] = np.ones_like(data[feature + '_sd'])
        if feature == 'compact':
            values = np.concatenate([values, np.broadcast_to(np.eye(4)[None, :, None, :],
                                     (len(ids), 4, 24, 4))], -1).astype('float32')
        data[feature] = values
        if feature in ['rich', 'compact']:
            data[feature + '_knots'] = np.quantile(values[:nt].reshape(-1, values.shape[-1]),
                                                  [.25, .5, .75], axis=0).astype('float32')
    np.savez(folder / 'data.npz', **data)
    meta = json.loads((folder / 'manifest.json').read_text())
    meta['preprocessing'] = 'fixed physical units, clip15, fitting-only additive knots'
    p.write(folder / 'manifest.json', meta)
    return folder


def save_recipe(folder, cid, quantiles):
    path = folder / 'trials' / str(cid)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'quantiles.npy', quantiles)


def blend_menu(members, objective, kind, prior=None):
    names = sorted(members)
    menu = [{name: 1.} for name in names]
    primary = 'fixed_linear_rich_' + objective
    assert primary in members
    menu += [{primary: w, other: 1-w} for other in names if other != primary for w in [.25, .5, .75]]
    top = sorted(names, key=lambda n: (members[n]['selection_ratio'], n))
    menu += [{name: 1/k for name in top[:k]} for k in [2, 3, 4, 6] if len(top) >= k]
    if kind == 'combined':
        menu += [{name: 1.} for name in prior]
    return menu


def select(folder, rows):
    d = np.load(folder / 'data.npz'); nt, ns = int(d['nt']), int(d['ns'])
    truth = d['y'][nt:nt+ns]; denom = float(d['selection_interpolation_mae'])
    winners = {}; predictions = {}
    for family, objective in itertools.product(p.FAMILIES, ['uniform', 'ratio']):
        name = f'fixed_{family}_{objective}'
        candidates = [dict(r, components={str(r['config']['id']): 1.}) for r in rows
                      if r['config']['family'] == family and r['config']['objective'] == objective]
        for lr in sorted({r['config']['lr'] for r in candidates}):
            pair = [r for r in candidates if r['config'].get('lr') == lr]
            # Smoke contains one seed, so its single is already the full recipe.
            if len(pair) != 2:
                continue
            parts = {str(r['config']['id']): .5 for r in pair}
            allq = sum(np.load(folder / 'trials' / cid / 'quantiles.npy') * w for cid, w in parts.items())
            cid = f'seedmean_{family}_{objective}_{lr:g}'
            save_recipe(folder, cid, allq)
            mae = float(np.abs(allq[:ns, ..., 5] - truth).mean())
            candidates.append(dict(config=dict(id=cid, family=family, objective=objective),
                                   components=parts, validation_mae=mae, selection_ratio=mae/denom))
        chosen = min(candidates, key=lambda r: (r['selection_ratio'], str(r['config']['id'])))
        winners[name] = dict(chosen, seed_candidates=candidates)
        predictions[name] = np.load(folder / 'trials' / str(chosen['config']['id']) / 'quantiles.npy')
    old_folder = OLD / 'folds' / folder.name
    old_data = np.load(old_folder / 'data.npz')
    for key in ['dates', 'y', 'h', 'b', 'nt', 'ns', 'nc', 'ne']:
        np.testing.assert_array_equal(d[key], old_data[key])
    old_selection = json.loads((old_folder / 'selection.json').read_text())
    names = [f'{f}_{o}' for f, o in itertools.product(p.FAMILIES, ['uniform', 'ratio'])]
    names += ['blend_uniform', 'blend_ratio']
    for original in names:
        name = 'standard_' + original; row = old_selection['by_recipe'][original]
        src = old_folder / 'trials' / str(row['config']['id']) / 'quantiles.npy'
        cid = name; allq = np.load(src); save_recipe(folder, cid, allq)
        # Only the common preceding selection block is used in this campaign.
        mae = float(np.abs(allq[:ns, ..., 5] - truth).mean())
        winners[name] = dict(config=dict(id=cid, family='frozen_standard', objective=row['config']['objective']),
                             validation_mae=mae, selection_ratio=mae/denom,
                             source=str(src.relative_to(PROJECT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                             original_recipe=original)
        predictions[name] = allq
    for objective, kind in itertools.product(['uniform', 'ratio'], ['fixed', 'combined']):
        members = {n: r for n, r in winners.items() if r['config']['objective'] == objective and
                   r['config']['family'] != 'blend' and 'blend_' not in n and
                   (kind == 'combined' or n.startswith('fixed_'))}
        prior = ['standard_blend_' + objective, 'fixed_blend_' + objective]
        menu = blend_menu(members, objective, kind, prior)
        choices = []
        for weights in menu:
            selected = sum(predictions[n][:ns] * w for n, w in weights.items())
            choices.append(dict(weights=weights, selection_ratio=float(np.abs(selected[..., 5]-truth).mean())/denom))
        chosen = min(choices, key=lambda r: r['selection_ratio'])
        name = f'{kind}_blend_{objective}'
        allq = sum(predictions[n] * w for n, w in chosen['weights'].items())
        save_recipe(folder, name, allq); predictions[name] = allq
        winners[name] = dict(config=dict(id=name, family='blend', objective=objective),
                             selection_ratio=chosen['selection_ratio'], validation_mae=chosen['selection_ratio']*denom,
                             weights=chosen['weights'], blend_candidates=choices, kind=kind)
    empirical = np.quantile(d['y'][:nt]-d['b'][:nt], TAUS, axis=(0, 2)).transpose(1, 2, 0)
    empirical[..., :5] = np.minimum(empirical[..., :5], 0); empirical[..., 5] = 0
    empirical[..., 6:] = np.maximum(empirical[..., 6:], 0)
    save_recipe(folder, 'interpolation', d['b'][nt:, ..., None]+empirical[None, :, None, :, :])
    winners['interpolation'] = dict(config=dict(id='interpolation', family='interpolation'),
                                   selection_ratio=1., validation_mae=denom)
    chosen = min(winners, key=lambda n: (winners[n]['selection_ratio'], n))
    winners['validation_selector'] = dict(winners[chosen], chosen_recipe=chosen)
    p.write(folder / 'selection.json', dict(by_recipe=winners, candidates=rows))
    return winners


def run(smoke=False):
    p.setup(); raw = np.load(ROOT / 'raw.npz'); folds = p.folds(raw['dates'])
    if smoke:
        folds = [folds[0], folds[-1]]
    reports = []
    for fold in folds:
        folder = folder_data(fold, smoke)
        jobs = configs()
        if smoke:
            jobs = [c for c in jobs if c['lr'] == .001 and c['seed'] == 17]
        rows = []
        with cf.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
            for future in cf.as_completed([pool.submit(q.fit, c, str(folder), smoke) for c in jobs]):
                row = future.result(); rows.append(row)
                p.write(folder / 'progress.json', dict(completed=len(rows), total=len(jobs)))
                print('FIT', fold['test'], len(rows), len(jobs), row, flush=True)
        winners = select(folder, rows)
        reports.append(q.evaluate(folder, winners)); q.summarize(reports, smoke)
        p.write(ROOT / ('smoke_progress.json' if smoke else 'progress.json'),
                dict(completed_folds=len(reports), total_folds=len(folds), last_test_month=fold['test']))
        print('EXPANDING FOLD DONE', len(reports), len(folds), fold['test'], flush=True)
    out = ROOT / 'smoke' if smoke else ROOT
    pd.concat([pd.read_csv(out / ('' if smoke else 'folds') / f['test'] / 'predictions.csv.gz')
               for f in folds], ignore_index=True).to_csv(out / 'predictions.csv.gz', index=False,
                                                        compression='gzip', float_format='%.8f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run(args.stage == 'smoke')
