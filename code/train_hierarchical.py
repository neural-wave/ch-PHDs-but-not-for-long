"""Isolated Vast campaign: matched hierarchy/direct models and held-out conformal."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets, rich_features, baseline, ZONES
from hierarchical_models import HierarchicalNet, conformal_quantile
from hierarchical_features import lagged_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out', default='results/hierarchical_vast')
    p.add_argument('--hpo-trials', type=int, default=0)
    p.add_argument('--hpo-seed', type=int, default=20261016)
    p.add_argument('--parent-dir')
    args = p.parse_args()
    assert torch.cuda.is_available(), 'Remote GPU required'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates, y = read_targets('train_prices_15min.csv')
    assert len(dates) == 363 and dates[302] == '2026-08-01' and dates[317] == '2026-08-16' and dates[333] == '2026-09-01'
    h = y.mean(-1)
    b = baseline(h)
    rich = rich_features(dates,h,dates,y)
    inputs = {}
    def profile(lag_days=-1):
        if lag_days not in inputs:
            x = lagged_features(dates,h,dates,y,lag_days,rich).reshape(363,4,24,-1)
            mu = x[:302].mean((0,1,2))
            sd = x[:302].std((0,1,2)) + 1e-5
            xt = torch.tensor(np.clip((x-mu)/sd,-15,15),device='cuda')
            inputs[lag_days] = (x,mu,sd,xt)
        return inputs[lag_days]
    x,mu,sd,xt = profile()
    configs = [dict(hierarchical=hier, width=w, seed=s) for hier in [True, False] for w in [32, 64] for s in [17, 43]]
    if args.hpo_trials:
        rng = np.random.default_rng(args.hpo_seed)
        configs = []
        for i in range(args.hpo_trials):
            configs.append(dict(hierarchical=(i % 4 != 3), width=int(rng.choice([32,64,128])),
                blocks=int(rng.choice([1,2])), dropout=float(rng.choice([0.,.05,.1,.2])),
                lr=float(10**rng.uniform(-3.7,-2.5)), weight_decay=float(10**rng.uniform(-5,-1.3)),
                auxiliary=float(rng.choice([0.,.1,.25,.5,1.])), batch_size=int(rng.choice([16,32,64])),
                anchor=str(rng.choice(['flat','interpolation'])), seed=int(rng.integers(1,1000000)),
                lag_days=[0,1,3,7,14,28][(i//4)%6]))
    if args.smoke:
        configs = [configs[0], next(c for c in configs if not c['hierarchical'])]
    preds = [b]
    metadata = [dict(name='interpolation', selection_mae=float(np.abs(b[302:317] - y[302:317]).mean()))]
    if args.parent_dir:
        parent = Path(args.parent_dir)
        previous = json.loads((parent/'metrics.json').read_text())
        assert previous['source_sha256'] == hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()
        assert (previous['train_days'],previous['selection_days'],previous['calibration_days']) == (302,15,16)
        if previous['selected'] != 'interpolation':
            original = parent/(previous['selected']+'.pt')
            saved = torch.load(original,weights_only=False,map_location='cpu')
            cfg = saved['config']
            assert np.array_equal(mu,saved['mu']) and np.array_equal(sd,saved['sd'])
            frozen = HierarchicalNet(x.shape[-1],cfg['width'],cfg['hierarchical'],cfg.get('blocks',1),cfg.get('dropout',.1)).cuda().eval()
            frozen.load_state_dict(saved['state'])
            with torch.no_grad():
                raw = frozen(xt).cpu().numpy().astype(np.float64)
            anchor = b if cfg.get('anchor') == 'interpolation' else h[...,None]
            pp = anchor+20*(raw-raw.mean(-1,keepdims=True))
            score = float(np.abs(pp[302:317]-y[302:317]).mean())
            assert abs(score-min(m['selection_mae'] for m in previous['candidates']))<1e-4
            shutil.copyfile(original,out/'parent_selected.pt')
            preds.append(pp)
            metadata.append(dict(name='parent_selected',config=cfg,selection_mae=score,origin=str(original)))
    checkpoints = []
    for i, cfg in enumerate(configs):
        started = time.perf_counter()
        x,mu,sd,xt = profile(cfg.get('lag_days',-1))
        torch.manual_seed(cfg['seed'])
        anchor = b if cfg.get('anchor') == 'interpolation' else np.broadcast_to(h[...,None],y.shape)
        yt = torch.tensor((y-anchor)/20,dtype=torch.float32,device='cuda')
        net = HierarchicalNet(x.shape[-1], cfg['width'], cfg['hierarchical'],cfg.get('blocks',1),cfg.get('dropout',.1)).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.get('lr',.001), weight_decay=cfg.get('weight_decay',.01))
        best, best_epoch, state = float('inf'), 0, None
        for epoch in range(2 if args.smoke else (160 if args.hpo_trials else 220)):
            net.train()
            for inds in torch.randperm(302, device='cuda').split(cfg.get('batch_size',16)):
                q = net(xt[inds])
                loss = (q - yt[inds]).abs().mean()
                if cfg['hierarchical']:
                    loss = loss + cfg.get('auxiliary',.25) * (q.reshape(-1, 4, 24, 2, 2).mean(-1) - yt[inds].reshape(-1, 4, 24, 2, 2).mean(-1)).abs().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                opt.step()
            net.eval()
            with torch.no_grad():
                score = float((net(xt[302:317]) - yt[302:317]).abs().mean())
            if score < best:
                best, best_epoch = score, epoch + 1
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch + 1 - best_epoch >= (20 if args.hpo_trials else 30):
                break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            raw = net(xt).cpu().numpy().astype(np.float64)
        pred = anchor + 20 * (raw - raw.mean(-1, keepdims=True))
        assert np.isfinite(pred).all() and np.abs(pred.mean(-1) - h).max() < 1e-8
        path = out / f'candidate_{i}.pt'
        torch.save(dict(config=cfg, state=state, mu=mu, sd=sd, feature_count=x.shape[-1]), path)
        # Rebuild and replay on GPU before accepting serialized state.
        saved = torch.load(path, weights_only=False, map_location='cpu')
        replay = HierarchicalNet(x.shape[-1], cfg['width'], cfg['hierarchical'],cfg.get('blocks',1),cfg.get('dropout',.1)).cuda()
        replay.load_state_dict(saved['state'])
        replay.eval()
        with torch.no_grad():
            rr = replay(xt[333:]).cpu().numpy().astype(np.float64)
        rr = anchor[333:] + 20 * (rr - rr.mean(-1, keepdims=True))
        assert np.max(np.abs(rr - pred[333:])) < 1e-4
        metadata.append(dict(name=f'candidate_{i}', config=cfg, epochs=best_epoch, epochs_run=epoch+1, seconds=time.perf_counter()-started,
            selection_mae=float(np.abs(pred[302:317] - y[302:317]).mean())))
        preds.append(pred)
        checkpoints.append(str(path))
        (out / 'progress.json').write_text(json.dumps(metadata, indent=2))
        print(metadata[-1], flush=True)
    # Calibration targets never choose model, epoch, or ensemble weight.
    selected = int(np.argmin([m['selection_mae'] for m in metadata]))
    chosen = preds[selected]
    report = dict(source_sha256=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest(), smoke=args.smoke,
                  device=torch.cuda.get_device_name(), train_days=302, selection_days=15, calibration_days=16, test_days=30,
                  selected=metadata[selected]['name'], candidates=metadata, coverage_caveat='Time dependence/drift violate exchangeability; coverage is empirical, not unconditional guaranteed.',
                  max_hourly_mean_error=float(np.abs(chosen.mean(-1) - h).max()), serialized_replay='passed for every neural candidate', conformal=[])
    report['search'] = dict(method='seeded random search with validation early stopping' if args.hpo_trials else 'fixed grid',
        trials=len(configs), seed=args.hpo_seed, selection_metric='August1-15 quarter MAE')
    for m, pred in zip(metadata, preds):
        m['august_mae'] = float(np.abs(pred[302:333] - y[302:333]).mean())
        m['september_mae'] = float(np.abs(pred[333:] - y[333:]).mean())
        m['september_halfhour_mae'] = float(np.abs((pred[333:] - y[333:]).reshape(-1,4,24,2,2).mean(-1)).mean())
    report['test_mae'] = float(np.abs(chosen[333:] - y[333:]).mean())
    cal = np.abs(chosen[317:333] - y[317:333])
    testerr = np.abs(chosen[333:] - y[333:])
    bands = {}
    for alpha in [.5, .1, .05]:
        # Zone/quarter marginal bands; hours pooled within zone, quarter position.
        scores = cal.transpose(0,2,1,3).reshape(-1,4,4)
        q = conformal_quantile(scores, alpha)[None,:,None,:]
        # Max over four quarters: joint coverage of an hour within each zone.
        hscores = cal.max(-1).transpose(0,2,1).reshape(-1,4)
        hq = conformal_quantile(hscores, alpha)[None,:,None,None]
        for name, radius in [('marginal',q), ('hour_joint',hq)]:
            covered = testerr <= radius
            report['conformal'].append(dict(kind=name, nominal=1-alpha, radius=np.asarray(radius).ravel().tolist(),
                coverage=float(covered.mean()), coverage_by_zone=covered.mean((0,2,3)).tolist(),
                hour_joint_coverage=float(covered.all(-1).mean()), mean_width=float(2*radius.mean())))
            bands[f'{name}_{int((1-alpha)*100)}_lower'] = chosen[333:] - radius
            bands[f'{name}_{int((1-alpha)*100)}_upper'] = chosen[333:] + radius
    # One score per complete day protects all zones/hours jointly at nominal 90%.
    for alpha in [.5,.1]:
        dq = float(conformal_quantile(cal.max((1,2,3)), alpha))
        report['conformal'].append(dict(kind='day_joint', nominal=1-alpha, calibration_blocks=16, radius=dq,
            day_joint_coverage=float((testerr <= dq).all((1,2,3)).mean()), mean_width=2*dq))
        bands[f'day_joint_{round((1-alpha)*100)}_lower'], bands[f'day_joint_{round((1-alpha)*100)}_upper'] = chosen[333:] - dq, chosen[333:] + dq
    np.savez_compressed(out / 'test_predictions.npz', dates=dates[333:].astype('U10'), predicted=chosen[333:], observed=y[333:], **bands)
    rows = [(dates[d], zone, hr+1, q+1, chosen[d,z,hr,q]) for d in range(333,363) for z,zone in enumerate(ZONES) for hr in range(24) for q in range(4)]
    frame = pd.DataFrame(rows, columns=['delivery_date','market_id','hour_index','quarter_index','predicted_price_eur_mwh'])
    frame.to_csv(out / 'test_predictions.csv', index=False)
    loaded = pd.read_csv(out / 'test_predictions.csv')
    assert len(loaded)==11520 and not loaded.duplicated(list(frame.columns[:4])).any()
    assert np.max(np.abs(loaded.predicted_price_eur_mwh.to_numpy().reshape(30,4,24,4).mean(-1)-h[333:]))<1e-4
    report['csv_audit']='passed: 11520 finite unique rows and hourly means'
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    hourly = pd.DataFrame([(dates[d], zone, hr+1, h[d,z,hr]) for d in range(333,363) for z,zone in enumerate(ZONES) for hr in range(24)],
        columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh'])
    hourly.to_csv(out / 'qa_hourly.csv', index=False)
    subprocess.run([sys.executable, 'code/predict_hierarchical.py', '--model-dir', str(out), '--hourly', str(out/'qa_hourly.csv'),
        '--history', 'train_prices_15min.csv', '--output', str(out/'qa_replay.csv'), '--device', 'cuda'], check=True)
    replay_csv = pd.read_csv(out / 'qa_replay.csv')
    delta = float(np.max(np.abs(replay_csv.predicted_price_eur_mwh.to_numpy().reshape(30,4,24,4) - chosen[333:])))
    assert delta < 1e-4, delta
    interval_csv = pd.read_csv(out / 'qa_replay_intervals.csv')
    assert np.isfinite(interval_csv.select_dtypes(include='number')).all().all()
    for key, value in bands.items():
        assert np.max(np.abs(interval_csv[key].to_numpy().reshape(30,4,24,4) - value)) < 1e-4
    report['offline_point_and_interval_replay_max_error'] = delta
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('DONE', report['selected'], report['test_mae'], flush=True)


if __name__ == '__main__':
    main()
