"""Focused paired monthly-ratio optimization. All fitting requires remote CUDA."""
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE / 'sources'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from reconstruction import read_targets, rich_features, baseline
from interpretable_models import compact_features
from tune_parallel_cnn import make_features, Small2D, dimensions
from simple_search import model as simple_model

ROOT = PROJECT / 'results/monthly_ratio_20261017'
SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
FAMILIES = ['linear_rich', 'linear_compact', 'additive_compact', 'mlp_compact', 'cnn_rich', 'cnn2d_analog']
DEPENDENCIES = ['campaigns/monthly_ratio_20261017/sources/' + name for name in
                ['reconstruction.py', 'interpretable_models.py', 'interpolation_history.py',
                 'simple_search.py', 'tune_parallel_cnn.py', 'parallel_zone_cnn.py']] + [
                'campaigns/monthly_ratio_20261017/experiment.py',
                'campaigns/monthly_ratio_20261017/audit.py',
                'campaigns/monthly_ratio_20261017/run.py',
                'campaigns/monthly_ratio_20261017/PROTOCOL.md']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(seed=17):
    assert torch.cuda.is_available(), 'Run all experiment fitting on the authorized research GPU'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def configs():
    return [dict(id=i, family=f, objective=o, lr=lr, seed=s)
            for i, (f, o, lr, s) in enumerate(itertools.product(FAMILIES, ['uniform', 'ratio'], [.001, .003], [17, 43]))]


def folds(dates):
    months = sorted(set(str(d)[:7] for d in dates))
    assert len(months) == 12
    return [dict(number=i-1, train_start=months[0], train_end=months[i],
                 validation=months[i+1], test=months[i+2]) for i in range(2, 10)]


def weights_for(dates, y, b, mask, objective):
    """Compute weights from fitting dates only; mean1 changes just a common scale."""
    indices = np.flatnonzero(mask)
    months = np.asarray([str(d)[:7] for d in dates])[indices]
    errors = np.abs(y[indices] - b[indices]).mean((1, 2, 3))
    weights = np.ones(len(indices), dtype=np.float64)
    details = {}
    for month in sorted(set(months)):
        ids = months == month
        denom = float(errors[ids].mean())
        assert denom > 0
        details[month] = dict(days=int(ids.sum()), interpolation_mae=denom)
        if objective == 'ratio':
            weights[ids] = 1 / (int(ids.sum()) * denom)
    weights /= weights.mean()
    return weights, details


def weighting_test():
    dates = np.asarray(['2025-10-01', '2025-10-02', '2025-11-01'])
    b = np.zeros((3, 1, 1, 1));y = np.asarray([2., 2., 10.]).reshape(3, 1, 1, 1)
    w, _ = weights_for(dates, y, b, np.ones(3, bool), 'ratio')
    e = np.asarray([1., 3., 10.])
    relative = ((1 + 3) / 2 / 2 + 10 / 10) / 2
    raw = np.asarray([1 / 4, 1 / 4, 1 / 10])
    np.testing.assert_allclose(np.mean(w * e) * raw.mean() * len(dates) / 2, relative)
    assert np.isclose(relative, 1.0)


def features(dates, h, hd, hy):
    rich = rich_features(dates, h, hd, hy).reshape(len(dates), 4, 24, -1)
    cnn = make_features(dates, h, hd, hy, 'context')
    analog = cnn[:, -4:].reshape(len(dates), 4, 24, 4).astype('float64') * 20
    analog -= analog.mean(-1, keepdims=True)
    compact = compact_features(dates, h, hd, hy, True, interpolation_days=7)[0]
    curve = np.broadcast_to(h.reshape(len(dates), 1, 1, 96) / 100, (len(dates), 4, 24, 96))
    compact = np.concatenate([compact, analog / 20, curve], axis=-1)
    return dict(rich=rich.astype('float32'), compact=compact.astype('float32'),
                cnn=cnn, analog=analog, curve=(h.reshape(len(dates), 96) / 100).astype('float32'))


def prepare():
    setup();weighting_test()
    ROOT.mkdir(parents=True, exist_ok=True)
    source = PROJECT / 'train_prices_15min.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SHA
    dates, y = read_targets(source);dates = np.asarray(dates, dtype=object);h = y.mean(-1);b = baseline(h)
    assert len(dates) == 363
    raw = features(dates, h, dates, y)
    for index in [0, 91, 150, 302, 333]:
        changed = y.copy();changed[index:] += np.arange(4) * 123456
        probe = features(dates[index:index+1], h[index:index+1], dates, changed)
        truncated = features(dates[index:index+1], h[index:index+1], dates[:index], y[:index])
        for key in raw:
            np.testing.assert_allclose(raw[key][index:index+1], probe[key], rtol=0, atol=1e-9)
            np.testing.assert_allclose(raw[key][index:index+1], truncated[key], rtol=0, atol=1e-9)
    hashes = {p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() for p in DEPENDENCIES}
    manifest = dict(source_sha256=SHA, code_hashes=hashes, configs=configs(), folds=folds(dates),
                    raw_shapes={k: list(v.shape) for k, v in raw.items()},
                    causal_probes=[str(dates[i]) for i in [0, 91, 150, 302, 333]],
                    raw_feature_causality_passed=True, monthly_weight_fixture_passed=True)
    if (ROOT / 'manifest.json').exists():
        assert json.loads((ROOT / 'manifest.json').read_text()) == manifest, 'Existing experiment differs; use a new output directory'
    np.savez(ROOT / 'raw.npz', dates=dates.astype('U10'), y=y, h=h, b=b, **raw)
    write(ROOT / 'manifest.json', manifest)
    print('PREPARED', manifest['raw_shapes'], flush=True)


def fold_data(fold, smoke=False):
    d = np.load(ROOT / 'raw.npz');dates = d['dates'];months = np.asarray([x[:7] for x in dates])
    day = np.asarray([int(x[8:]) for x in dates])
    train = (months >= fold['train_start']) & (months <= fold['train_end'])
    select = (months == fold['validation']) & (day <= 15)
    cal = (months == fold['validation']) & (day > 15);test = months == fold['test']
    ids = np.flatnonzero(train | select | cal | test)
    nt, ns, nc, ne = (int(v.sum()) for v in [train, select, cal, test])
    assert ns == 15 and max(dates[train]) < min(dates[select]) < min(dates[cal]) < min(dates[test])
    folder = ROOT / ('smoke' if smoke else 'folds') / fold['test']
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(dates=dates[ids], y=d['y'][ids], h=d['h'][ids], b=d['b'][ids],
                  analog=d['analog'][ids], nt=nt, ns=ns, nc=nc, ne=ne)
    stats = {}
    for key in ['rich', 'compact', 'cnn', 'curve']:
        raw = d[key];axes = tuple(range(raw.ndim - 1)) if key in ['rich', 'compact'] else (0, 2) if key == 'cnn' else (0,)
        mu = raw[train].mean(axes, keepdims=True);sd = raw[train].std(axes, keepdims=True).clip(.05)
        xx = np.clip((raw[ids] - mu) / sd, -15, 15).astype('float32')
        if key == 'compact':
            xx = np.concatenate([xx, np.broadcast_to(np.eye(4)[None, :, None, :], (len(ids), 4, 24, 4))], -1).astype('float32')
        arrays[key] = xx;arrays[key + '_mu'] = mu;arrays[key + '_sd'] = sd
        if key in ['rich', 'compact']:
            arrays[key + '_knots'] = np.quantile(xx[:nt].reshape(-1, xx.shape[-1]), [.25, .5, .75], axis=0).astype('float32')
        stats[key] = dict(axes=axes, shape=list(xx.shape))
    for objective in ['uniform', 'ratio']:
        weights, details = weights_for(dates, d['y'], d['b'], train, objective)
        arrays['weights_' + objective] = weights.astype('float32')
    assert all(m <= fold['train_end'] for m in details)
    arrays['selection_interpolation_mae'] = float(np.abs(d['y'][select] - d['b'][select]).mean())
    np.savez(folder / 'data.npz', **arrays)
    meta = dict(**fold, train_days=nt, selection_days=ns, calibration_days=nc, test_days=ne,
                training_month_baselines=details, normalization=stats,
                boundaries=[str(min(dates[v])) for v in [train, select, cal, test]] + [str(max(dates[test]))])
    write(folder / 'manifest.json', meta)
    return folder


class Global2D(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = dict(width=16, clock=True, residual=True, analog=90, context=5)
        base = Small2D(cfg)
        self.channels, self.aux = dimensions(cfg)
        self.conv, self.fusion, self.output = base.conv, base.fusion, base.output
        self.full_curve = nn.Linear(96, 16)

    def forward(self, x, curve):
        grid = x[:, :-self.aux].reshape(-1, self.channels, 16, 96)
        encoded = F.relu(self.conv(F.pad(grid, (3, 4, 0, 0), mode='replicate'))).flatten(1, 2)
        fused = F.relu(self.fusion(torch.cat([encoded, x[:, -self.aux:]], 1)) + self.full_curve(curve)[:, :, None])
        out = self.output(fused).reshape(-1, 4, 24, 4)
        return out - out.mean(-1, keepdim=True)


def network(cfg, data):
    family = cfg['family']
    if family == 'cnn2d_analog':
        return Global2D(), 'cnn'
    base, feature = family.split('_', 1)
    return simple_model(dict(family=base, width=32 if base == 'mlp' else 16,
                             depth=2 if base == 'mlp' else 5), data[feature].shape[-1], data[feature + '_knots']), feature


def predict(net, cfg, x, curve):
    return net(x, curve) if cfg['family'] == 'cnn2d_analog' else net(x)


def fit(cfg, folder, smoke=False):
    setup(cfg['seed']);start = time.time()
    folder = Path(folder);dest = folder / 'trials' / str(cfg['id'])
    if (dest / 'metrics.json').exists() and (dest / 'predictions.npy').exists():
        result = json.loads((dest / 'metrics.json').read_text())
        assert result['config'] == cfg
        return result
    d = np.load(folder / 'data.npz');nt, ns = int(d['nt']), int(d['ns'])
    net, feature = network(cfg, d);net = net.cuda()
    # Only train and selection targets reach training/early stopping.
    x = torch.as_tensor(d[feature][:nt+ns], device='cuda')
    curve = torch.as_tensor(d['curve'][:nt+ns], device='cuda')
    target = torch.tensor((d['y'][:nt+ns] - d['b'][:nt+ns]) / 20, dtype=torch.float32, device='cuda')
    weights = torch.as_tensor(d['weights_' + cfg['objective']], device='cuda')
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=.01)
    limit = 2 if smoke else 450 if cfg['family'] == 'cnn2d_analog' else 300
    patience = 50 if cfg['family'] == 'cnn2d_analog' else 35
    best, best_epoch, state = float('inf'), 0, None
    curves = []
    for epoch in range(1, limit+1):
        net.train()
        for ids in torch.randperm(nt, device='cuda').split(32):
            daily = (predict(net, cfg, x[ids], curve[ids]) - target[ids]).abs().mean((1, 2, 3))
            loss = (daily * weights[ids]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():
            value = float((predict(net, cfg, x[nt:], curve[nt:]) - target[nt:]).abs().mean()) * 20
        assert np.isfinite(value)
        curves.append([epoch, value / float(d['selection_interpolation_mae'])])
        if value < best:
            best, best_epoch = value, epoch
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    net.load_state_dict(state);net.eval()
    ex = torch.as_tensor(d[feature][nt:], device='cuda');ec = torch.as_tensor(d['curve'][nt:], device='cuda')
    with torch.no_grad():
        raw = predict(net, cfg, ex, ec).cpu().numpy().astype('float64')
    pred = d['b'][nt:] + 20 * (raw - raw.mean(-1, keepdims=True))
    assert np.isfinite(pred).all() and np.abs(pred.mean(-1) - d['h'][nt:]).max() < 1e-8
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=cfg, state=state, feature=feature), dest / 'model.pt')
    np.save(dest / 'predictions.npy', pred)
    np.savetxt(dest / 'curves.csv', curves, delimiter=',', header='epoch,selection_ratio', comments='')
    replay, _ = network(cfg, d);replay.cuda().eval().load_state_dict(torch.load(dest / 'model.pt', weights_only=False, map_location='cpu')['state'])
    with torch.no_grad():
        rr = predict(replay, cfg, ex, ec).cpu().numpy().astype('float64')
    delta = float(np.max(np.abs(rr - raw)));assert delta < 1e-7
    val = float(np.abs(pred[:ns] - d['y'][nt:nt+ns]).mean())
    result = dict(config=cfg, validation_mae=val, selection_ratio=val / float(d['selection_interpolation_mae']),
                  parameters=sum(p.numel() for p in net.parameters()), best_epoch=best_epoch, epochs=epoch,
                  seconds=time.time()-start, replay_max_error=delta)
    write(dest / 'metrics.json', result)
    return result


def pool(folder, smoke=False):
    jobs = configs()
    if smoke:
        jobs = [next(c for c in jobs if c['family'] == f and c['objective'] == o) for f, o in itertools.product(FAMILIES, ['uniform', 'ratio'])]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(fit, cfg, str(folder), smoke) for cfg in jobs]
        for future in cf.as_completed(futures):
            result = future.result();rows.append(result)
            write(folder / 'progress.json', dict(completed=len(rows), total=len(jobs), latest=result))
            print('FIT', folder.name, len(rows), len(jobs), result, flush=True)
    return rows


def controls(folder, rows, smoke=False):
    d = np.load(folder / 'data.npz');nt, ns = int(d['nt']), int(d['ns'])
    def save(cid, family, objective, pred, parameters=0, **extra):
        path = folder / 'trials' / cid;path.mkdir(parents=True, exist_ok=True)
        np.save(path / 'predictions.npy', pred)
        val = float(np.abs(pred[:ns] - d['y'][nt:nt+ns]).mean())
        row = dict(config=dict(id=cid, family=family, objective=objective, **extra), parameters=parameters,
                   validation_mae=val, selection_ratio=val / float(d['selection_interpolation_mae']))
        rows.append(row);write(path / 'metrics.json', row)
        return path
    for feature, objective in itertools.product(['rich', 'compact'], ['uniform', 'ratio']):
        xx = d[feature].astype('float64').reshape(len(d['dates']) * 96, -1)
        x = np.column_stack([np.ones(len(xx)), xx])
        target = ((d['y'] - d['b']) / 20).reshape(-1, 4)
        w = np.repeat(d['weights_' + objective].astype('float64'), 96)
        gram = x[:nt*96].T @ (x[:nt*96] * w[:, None]);cross = x[:nt*96].T @ (target[:nt*96] * w[:, None])
        for alpha in ([1000] if smoke else [1, 10, 100, 1000, 10000]):
            penalty = np.eye(x.shape[-1]) * alpha;penalty[0, 0] = 0
            beta = np.linalg.solve(gram + penalty, cross)
            rr = (x[nt*96:] @ beta).reshape(-1, 4, 24, 4)
            pp = d['b'][nt:] + 20 * (rr - rr.mean(-1, keepdims=True))
            path = save(f'ridge_{feature}_{objective}_{alpha}', 'ridge_' + feature, objective, pp, beta.size, alpha=alpha)
            np.save(path / 'beta.npy', beta)
    for scale in [0, .25, .5, .75, 1]:
        save('analog_' + str(scale), 'analog_shrink', 'fixed', d['b'][nt:] + scale * d['analog'][nt:], scale=scale)
    save('interpolation', 'interpolation', 'fixed', d['b'][nt:])
    return rows


def selections(folder, rows):
    d = np.load(folder / 'data.npz');nt, ns = int(d['nt']), int(d['ns'])
    groups = sorted({(r['config']['family'], r['config']['objective']) for r in rows})
    winners = {}
    for family, objective in groups:
        members = [r for r in rows if (r['config']['family'], r['config']['objective']) == (family, objective)]
        winners[family + '_' + objective] = min(members, key=lambda r: (r['selection_ratio'], str(r['config']['id'])))
    for objective in ['uniform', 'ratio']:
        members = {name: row for name, row in winners.items() if row['config']['objective'] == objective}
        preds = {name: np.load(folder / 'trials' / str(row['config']['id']) / 'predictions.npy') for name, row in members.items()}
        primary = 'linear_rich_' + objective
        blends = [{name: 1.0} for name in sorted(members)]
        blends += [{primary: weight, other: 1-weight} for other in sorted(members) if other != primary for weight in [.25, .5, .75]]
        top = sorted(members, key=lambda name: (members[name]['selection_ratio'], name))[:3]
        blends.append({name: 1/3 for name in top})
        scored = []
        for weights in blends:
            pred = sum(preds[name][:ns] * weight for name, weight in weights.items())
            ratio = float(np.abs(pred - d['y'][nt:nt+ns]).mean()) / float(d['selection_interpolation_mae'])
            scored.append(dict(weights=weights, selection_ratio=ratio))
        chosen = min(scored, key=lambda x: x['selection_ratio']);cid = 'blend_' + objective
        path = folder / 'trials' / cid;path.mkdir(parents=True, exist_ok=True)
        np.save(path / 'predictions.npy', sum(preds[name] * weight for name, weight in chosen['weights'].items()))
        winners[cid] = dict(config=dict(id=cid, family='blend', objective=objective),
                            parameters=sum(members[name]['parameters'] for name in chosen['weights']),
                            validation_mae=chosen['selection_ratio'] * float(d['selection_interpolation_mae']),
                            selection_ratio=chosen['selection_ratio'], weights=chosen['weights'], blend_candidates=scored)
    # This deployable rule selects from validation results only, including the two blends.
    chosen = min(winners, key=lambda name: (winners[name]['selection_ratio'], name))
    winners['validation_selector'] = dict(winners[chosen], chosen_recipe=chosen)
    # Selection is written before any test scoring.
    write(folder / 'selection.json', dict(by_recipe=winners, candidates=rows))
    return winners


def score(folder, winners):
    d = np.load(folder / 'data.npz');nt, ns, nc = (int(d[k]) for k in ['nt', 'ns', 'nc'])
    y = d['y'][nt+ns+nc:];b = d['b'][nt+ns+nc:];base = float(np.abs(b-y).mean())
    results = {}
    for name, row in winners.items():
        pred = np.load(folder / 'trials' / str(row['config']['id']) / 'predictions.npy')[ns+nc:]
        err = np.abs(pred-y);mean_error = float(np.abs(pred.mean(-1)-d['h'][nt+ns+nc:]).max())
        assert np.isfinite(pred).all() and mean_error < 1e-8
        results[name] = dict(mae=float(err.mean()), interpolation_mae=base, ratio=float(err.mean()) / base,
                             by_zone=err.mean((0, 2, 3)).tolist(),
                             ratio_by_zone=(err.mean((0, 2, 3))/np.abs(b-y).mean((0, 2, 3))).tolist(),
                             daily_mae=err.mean((1, 2, 3)).tolist(), max_hourly_mean_error=mean_error)
    report = dict(manifest=json.loads((folder/'manifest.json').read_text()), selected=winners, results=results)
    write(folder / 'metrics.json', report)
    return report


def summarize(reports, smoke=False):
    out = ROOT / 'smoke' if smoke else ROOT
    rows = []
    for report in reports:
        meta = report['manifest']
        for name, result in report['results'].items():
            rows.append(dict(month=meta['test'], recipe=name, train_days=meta['train_days'], test_days=meta['test_days'],
                             mae=result['mae'], interpolation_mae=result['interpolation_mae'], ratio=result['ratio'],
                             selection_ratio=report['selected'][name]['selection_ratio'],
                             **{f'mae_ZONE_{z}': result['by_zone'][z] for z in range(4)},
                             **{f'ratio_ZONE_{z}': result['ratio_by_zone'][z] for z in range(4)}))
    frame = pd.DataFrame(rows);frame.to_csv(out / 'monthly_scores.csv', index=False)
    summary = []
    for name, group in frame.groupby('recipe'):
        summary.append(dict(recipe=name, folds=len(group), mean_monthly_ratio=float(group.ratio.mean()),
                            pooled_mae=float(np.average(group.mae, weights=group.test_days)),
                            worst_ratio=float(group.ratio.max()), months_beating_interpolation=int((group.ratio < 1-1e-12).sum())))
    summary.sort(key=lambda r: r['mean_monthly_ratio'])
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    write(out / 'metrics.json', dict(smoke=smoke, completed_folds=len(reports), summary=summary,
                                    caveat='Developmental backtests; unequal candidate spaces; outer test scores never select a candidate within a fold.'))
    return summary


def run(smoke=False):
    setup();d = np.load(ROOT / 'raw.npz');jobs = folds(d['dates'])
    if smoke:
        jobs = [jobs[0], jobs[-1]]
    reports = []
    for fold in jobs:
        folder = fold_data(fold, smoke)
        rows = controls(folder, pool(folder, smoke), smoke)
        winners = selections(folder, rows);reports.append(score(folder, winners))
        summary = summarize(reports, smoke)
        write(ROOT / ('smoke_progress.json' if smoke else 'progress.json'),
              dict(stage='smoke' if smoke else 'full fitting', completed_folds=len(reports), total_folds=len(jobs),
                   last_test_month=fold['test'], latest_best=summary[0]))
        print('FOLD DONE', len(reports), len(jobs), fold['test'], summary[0], flush=True)
    if smoke:
        # Nonzero global conditioning gradient, independent of zero initial head.
        folder = ROOT / 'smoke' / jobs[0]['test'];data = np.load(folder/'data.npz')
        net = Global2D().cuda();nn.init.normal_(net.output.weight, std=.1)
        curve = torch.tensor(data['curve'][:2], device='cuda', requires_grad=True)
        x = torch.as_tensor(data['cnn'][:2], device='cuda')
        net(x, curve).square().sum().backward()
        assert torch.isfinite(curve.grad).all() and (curve.grad.abs().sum(0) > 0).all()
        write(ROOT/'smoke.json', dict(passed=True, folds=2, neural_fits=24, objectives=['uniform', 'ratio'],
                                     monthly_weight_fixture=True, causal_features=True, full_hourly_conditioning_gradient=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        run(args.stage == 'smoke')
