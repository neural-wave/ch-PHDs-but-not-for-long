"""Small paired-row 2D CNN: constrained MAE decision and ordered marginal quantiles."""
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reconstruction import read_targets, baseline
from tune_parallel_cnn import make_features as existing_features, dimensions, ARMS

ROOT = Path('results/quantile_2d_cnn_20261017')
SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
TAUS = np.array([.025, .05, .10, .20, .25, .50, .75, .80, .90, .95, .975])
PACKS = {'raw': 'clock', 'analog90': 'analog90'}
TRAIN_END, SELECT_END, CAL_END = 302, 317, 333
OOF_EDGES = [120, 151, 181, 212, 242, 272, 302]
SEVERE_MAE = 10.0


def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def configs():
    return [dict(id=i, pack=p, quantile_weight=q, lr=lr, seed=s, width=16)
            for i, (p, q, lr, s) in enumerate(itertools.product(PACKS, [0., 1.], [.001, .003], [17, 43, 71]))]


def layout(pack):
    return dimensions(ARMS[PACKS[pack]])


def make_features(dates, hourly, history_dates, history_y, pack):
    return existing_features(dates, hourly, history_dates, history_y, PACKS[pack])


def normalize(raw, mu, sd, pack):
    x = np.clip((raw - mu) / sd, -15, 15).copy()
    ch, aux = layout(pack)
    grid = x[:, :-aux].reshape(-1, ch, 16, 96)
    observed = raw[:, :-aux].reshape(-1, ch, 16, 96)[:, 4:5]
    grid[:, :4] *= observed
    grid[:, 4:5] = observed  # A literal 0/1 availability mask, never standardized.
    if ch == 9:
        grid[:, 5:] *= observed
    return x


class QuantileCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ch, self.aux = layout(cfg['pack'])
        w = cfg['width']
        self.conv = nn.Conv2d(self.ch, w, (2, 8), stride=(2, 1))
        self.fusion = nn.Conv1d(w * 8 + self.aux, w, 1)
        self.point = nn.Conv1d(w, 4, 1)
        self.median = nn.Conv1d(w, 4, 1)
        self.increments = nn.Conv1d(w, 40, 1)
        for layer in [self.point, self.median, self.increments]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        grid = x[:, :-self.aux].reshape(-1, self.ch, 16, 96)
        conv = F.relu(self.conv(F.pad(grid, (3, 4, 0, 0), mode='replicate')))
        fused = F.relu(self.fusion(torch.cat([conv.flatten(1, 2), x[:, -self.aux:]], 1)))
        point = self.point(fused).reshape(-1, 4, 24, 4)
        point = point - point.mean(-1, keepdim=True)
        median = self.median(fused).reshape(-1, 4, 24, 4, 1)
        inc = F.softplus(self.increments(fused).reshape(-1, 4, 10, 96)
                         .permute(0, 1, 3, 2).reshape(-1, 4, 24, 4, 10))
        low = (median - inc[..., :5].cumsum(-1)).flip(-1)
        high = median + inc[..., 5:].cumsum(-1)
        return point, torch.cat([low, median, high], -1), conv, fused

    def initialize_quantiles(self, targets):
        empirical = np.quantile(targets, TAUS, axis=(0, 2, 3)).T
        gaps = np.concatenate([np.diff(empirical[:, :6], axis=1)[:, ::-1],
                               np.diff(empirical[:, 5:], axis=1)], 1)
        gaps = np.maximum(gaps, .005)
        with torch.no_grad():
            self.median.bias.copy_(torch.as_tensor(empirical[:, 5], dtype=self.median.bias.dtype))
            self.increments.bias.copy_(torch.as_tensor(np.log(np.expm1(gaps)).ravel(),
                                                       dtype=self.increments.bias.dtype))


def objective(point, quantiles, target, weight):
    mae = (point - target).abs().mean()
    tau = torch.as_tensor(TAUS, dtype=target.dtype, device=target.device)
    err = target[..., None] - quantiles
    pinball = torch.maximum(tau * err, (tau - 1) * err).mean()
    return mae + weight * pinball, mae, pinball


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest() == SHA
    dates, y = read_targets('train_prices_15min.csv')
    assert len(dates) == 363 and dates[302] == '2026-08-01' and dates[333] == '2026-09-01'
    h = y.mean(-1)
    np.savez(ROOT / 'data.npz', dates=dates.astype('U10'), y=y, h=h, b=baseline(h))
    for pack in PACKS:
        raw = make_features(dates, h, dates, y, pack)
        mu = raw[:TRAIN_END].mean((0, 2), keepdims=True)
        sd = raw[:TRAIN_END].std((0, 2), keepdims=True).clip(.05)
        np.savez(ROOT / f'{pack}.npz', raw=raw, x=normalize(raw, mu, sd, pack), mu=mu, sd=sd)
    write_json(ROOT / 'manifest.json', dict(sha256=SHA, configs=configs(), taus=TAUS.tolist(),
        train=[0, TRAIN_END], selection=[TRAIN_END, SELECT_END], calibration=[SELECT_END, CAL_END],
        test=[CAL_END, 363], oof_edges=OOF_EDGES, severe_day_mae_threshold=SEVERE_MAE,
        seed_selection='August1-15 point MAE only; choices persisted before September inference',
        calibration_note='August16-31, separate from stopping and selection; no inference-time adaptation',
        architecture='Conv2d paired Q/H2x8,16 filters; lag-preserving1x1 fusion16; point4 and independent11 ordered quantiles',
        uncertainty='Only point decision conserves hourly means. Independent q50 need not equal constrained MAE decision.',
        history='7 calendar days raw Q/H; analog90 variant also uses fixed retrieval from strictly earlier90days',
        caveat='September developmental, previously examined; performance target5.2 is not a selection rule.'))


def fit(cfg, fit_end=TRAIN_END, val_end=SELECT_END, epochs=600, folder=None):
    configure()
    torch.manual_seed(cfg['seed'])
    start = time.perf_counter()
    d, f = np.load(ROOT / 'data.npz'), np.load(ROOT / f"{cfg['pack']}.npz")
    raw = f['raw']
    mu = raw[:fit_end].mean((0, 2), keepdims=True)
    sd = raw[:fit_end].std((0, 2), keepdims=True).clip(.05)
    x = torch.as_tensor(normalize(raw[:val_end], mu, sd, cfg['pack']), device='cuda')
    targets = (d['y'][:val_end] - d['b'][:val_end]) / 20
    target = torch.as_tensor(targets, dtype=torch.float32, device='cuda')
    net = QuantileCNN(cfg)
    net.initialize_quantiles(targets[:fit_end])
    net = net.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=.01)
    best, best_epoch, state, curves = float('inf'), 0, None, []
    for epoch in range(1, epochs + 1):
        net.train()
        totals = np.zeros(3)
        for ids in torch.randperm(fit_end, device='cuda').split(32):
            out = net(x[ids])
            loss, mae, pb = objective(out[0], out[1], target[ids], cfg['quantile_weight'])
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            totals += np.array([float(loss.detach()), float(mae.detach()), float(pb.detach())]) * len(ids)
        net.eval()
        with torch.no_grad():
            out = net(x[fit_end:val_end])
            _, mae, pb = objective(out[0], out[1], target[fit_end:val_end], cfg['quantile_weight'])
            val = float(mae) * 20
        assert np.isfinite(val)
        curves.append([epoch, *(totals / fit_end * 20).tolist(), val, float(pb) * 20])
        if val < best:
            best, best_epoch = val, epoch
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch - best_epoch >= 60:
            break
    meta = dict(config=cfg, fit_end=fit_end, val_end=val_end, validation_mae=best,
                best_epoch=best_epoch, epochs=epoch, seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in net.parameters()),
                point_parameters=sum(p.numel() for name, p in net.named_parameters()
                                     if not name.startswith(('median.', 'increments.'))))
    if folder is not None:
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=cfg, state=state, mu=mu, sd=sd, fit_end=fit_end, val_end=val_end), folder/'model.pt')
        write_json(folder/'metrics.json', meta)
        np.savetxt(folder/'learning_curves.csv', curves, delimiter=',', comments='',
                   header='epoch,train_joint,train_mae,train_pinball,selection_mae,selection_pinball')
    return meta


def run_trial(cfg):
    return fit(cfg, folder=ROOT / 'trials' / str(cfg['id']))


def load_model(folder):
    ck = torch.load(Path(folder)/'model.pt', map_location='cpu', weights_only=False)
    net = QuantileCNN(ck['config']).cuda().eval()
    net.load_state_dict(ck['state'])
    return net, ck


def predict(folder, start, end):
    configure()
    net, ck = load_model(folder)
    d, f = np.load(ROOT/'data.npz'), np.load(ROOT/f"{ck['config']['pack']}.npz")
    x = normalize(f['raw'][start:end], ck['mu'], ck['sd'], ck['config']['pack'])
    with torch.no_grad():
        out = net(torch.as_tensor(x, device='cuda'))
    point, q, conv, fused = [v.cpu().numpy().astype('float64') for v in out]
    point = d['b'][start:end] + 20*(point-point.mean(-1, keepdims=True))
    q = d['b'][start:end, ..., None] + 20*q
    assert np.max(np.abs(point.mean(-1)-d['h'][start:end])) < 1e-8
    assert np.diff(q, axis=-1).min() >= -1e-5
    return dict(point=point, quantiles=q, conv=conv, fused=fused)


def train_all():
    rows = []
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as ex:
        for fut in cf.as_completed([ex.submit(run_trial, cfg) for cfg in configs()]):
            rows.append(fut.result())
            write_json(ROOT/'progress.json', dict(stage='main fits', completed=len(rows), total=24, latest=rows[-1]))
            print('FIT', len(rows), 24, rows[-1], flush=True)
    groups = {f'{pack}_{int(weight)}': min((r for r in rows if r['config']['pack']==pack
              and r['config']['quantile_weight']==weight), key=lambda r:r['validation_mae'])
              for pack in PACKS for weight in [0., 1.]}
    selected = min(rows, key=lambda r:r['validation_mae'])
    quantile_selected = min((r for r in rows if r['config']['quantile_weight']>0), key=lambda r:r['validation_mae'])
    selection = dict(overall=selected, quantile=quantile_selected, by_group=groups, candidates=rows)
    write_json(ROOT/'selection.json', selection)  # Freeze before any September inference.
    reports = {}
    d = np.load(ROOT/'data.npz')
    for key, row in groups.items():
        folder = ROOT/'trials'/str(row['config']['id'])
        pred = predict(folder, TRAIN_END, len(d['dates']))
        np.savez(folder/'predictions.npz', **pred)
        errors = np.abs(pred['point'][CAL_END-TRAIN_END:]-d['y'][CAL_END:])
        reports[key] = dict(**row, september_mae=float(errors.mean()),
                            by_zone=errors.mean((0, 2, 3)).tolist(),
                            hourly_mean_error=float(np.abs(pred['point'].mean(-1)-d['h'][TRAIN_END:]).max()))
    report = dict(by_group=reports, selected_group=f"{selected['config']['pack']}_{int(selected['config']['quantile_weight'])}",
                  quantile_group=f"{quantile_selected['config']['pack']}_1",
                  interpolation_september_mae=float(np.abs(d['b'][CAL_END:]-d['y'][CAL_END:]).mean()),
                  caveat='August1-15 selection; August16-31 reserved calibration; September developmental.')
    write_json(ROOT/'metrics.json', report)
    print('POINT RESULTS', report, flush=True)


def oof_one(args):
    number, end, nxt, cfg = args
    folder = ROOT/'oof'/str(number)
    meta = fit(cfg, fit_end=end-15, val_end=end, folder=folder)
    out = predict(folder, end, nxt)
    np.savez(folder/'predictions.npz', start=end, end=nxt, **out)
    return dict(fold=number, prediction_start=end, prediction_end=nxt, **meta)


def oof():
    selected = json.loads((ROOT/'selection.json').read_text())['quantile']['config']
    jobs = [(i, a, b, selected) for i, (a, b) in enumerate(zip(OOF_EDGES[:-1], OOF_EDGES[1:]))]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('spawn')) as ex:
        for fut in cf.as_completed([ex.submit(oof_one, j) for j in jobs]):
            rows.append(fut.result())
            write_json(ROOT/'progress.json', dict(stage='chronological out-of-sample risk labels', completed=len(rows), total=6))
            print('OOF', rows[-1], flush=True)
    write_json(ROOT/'oof_manifest.json', dict(folds=sorted(rows,key=lambda r:r['fold']),
        note='Each checkpoint fits before its15day stopping block, predicts only later days. Main hyperparameters selected August1-15; not a nested historical model-selection benchmark.'))


def smoke():
    configure()
    assert torch.cuda.is_available()
    d = np.load(ROOT/'data.npz')
    checks = {}
    changed = d['y'].copy(); changed[333:] += np.arange(4)*12345
    for pack in PACKS:
        f = np.load(ROOT/f'{pack}.npz'); ch, aux = layout(pack)
        probe = make_features(d['dates'][333:334], d['h'][333:334], d['dates'], changed, pack)
        np.testing.assert_array_equal(probe, f['raw'][333:334])
        np.testing.assert_array_equal(probe, make_features(d['dates'][333:334], d['h'][333:334], d['dates'][:333], d['y'][:333], pack))
        grid = f['raw'][333:334,:-aux].reshape(1,ch,16,96)
        assert not grid[:,:4,14].any() and not grid[:,4,14].any() and grid[:,4,15].all()
        np.testing.assert_allclose(grid[0,:4,15], np.repeat(d['h'][333],4,axis=-1)/100,rtol=1e-6)
        missing = make_features(d['dates'][333:334],d['h'][333:334],[],[],pack)
        mg = normalize(missing,f['mu'],f['sd'],pack)[:,:-aux].reshape(1,ch,16,96)
        assert not mg[:,:,:14].any() and not mg[:,:,14].any()
        cfg = next(c for c in configs() if c['pack']==pack and c['quantile_weight']==1)
        net = QuantileCNN(cfg)
        target = (d['y'][:302]-d['b'][:302])/20
        net.initialize_quantiles(target)
        with torch.no_grad():
            net.median.bias.add_(.25)  # Deliberately noncentered to verify independence.
        net = net.cuda()
        nn.init.normal_(net.point.weight,std=.01)
        xx = torch.tensor(f['x'][302:304],device='cuda',requires_grad=True)
        point,q,_,_ = net(xx)
        point[:,0,:,0].sum().backward()
        assert xx.grad[:,31].abs().sum()>0  # Query hourly prices of zone1 affect zone0.
        assert point.mean(-1).abs().max()<1e-6 and torch.diff(q,dim=-1).min()>0
        # The independent q50 is not forcibly centered like the point head.
        assert q[...,5].mean(-1).abs().max()>1e-4
        err = target[...,None]-np.quantile(target,TAUS,axis=(0,2,3)).T[None,:,None,None,:]
        assert np.isfinite(err).all()
        meta = fit(cfg,epochs=2,folder=ROOT/'smoke'/pack)
        replay = predict(ROOT/'smoke'/pack,302,304)
        assert np.isfinite(replay['point']).all()
        checks[pack] = dict(causality=True,mask=True,query_hourly=True,conservation=True,ordered_quantiles=True,
                            independent_q50=True,cross_zone_gradient=True,two_epoch_fit=meta)
    write_json(ROOT/'smoke.json',dict(passed=True,gpu=torch.cuda.get_device_name(),checks=checks))
    print('SMOKE PASSED',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','oof']);a=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'train':train_all,'oof':oof}[a.stage]()


if __name__=='__main__':
    main()
