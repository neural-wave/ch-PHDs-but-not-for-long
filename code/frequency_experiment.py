"""Daily low-rank residual experiment; all selection uses August only."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

OUT = Path('results/four_frequency_20261017')


def project(x):
    shape = x.shape
    q = x.reshape(*shape[:-1], 24, 4)
    return (q - q.mean(-1, keepdims=True)).reshape(shape)


def make_bases(train):
    """Greedily select distinct fixed-phase frequencies using training SSE."""
    t = np.arange(96)
    dictionary = []
    for k in range(1, 49):
        for phase in ['cos', 'sin']:
            raw = getattr(np, phase)(2*np.pi*k*t/96)
            v = project(raw)
            if np.linalg.norm(v) > 1e-8:
                dictionary.append((k, phase, v))
    selected, labels, used = [], [], set()
    for _ in range(4):
        candidates = []
        for k, phase, raw in dictionary:
            if k in used:
                continue
            v = raw.copy()
            if selected:
                q = np.stack(selected)
                v -= (v @ q.T) @ q
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                v /= norm
                candidates.append((float(np.square(train @ v).sum()), k, phase, v))
        _, k, phase, v = max(candidates, key=lambda row: row[0])
        selected.append(v); labels.append(dict(cycles_per_day=k, phase=phase)); used.add(k)
    # QR spans original projected Fourier functions; coefficients can be converted.
    raw4 = np.stack([next(v for k, ph, v in dictionary if k == lab['cycles_per_day'] and ph == lab['phase']) for lab in labels])
    raw8 = np.stack([v for k, ph, v in dictionary if k in used])
    q8 = np.linalg.svd(raw8, full_matrices=False)[2][:np.linalg.matrix_rank(raw8)]
    _, _, pca = np.linalg.svd(train, full_matrices=False)
    bases = {'fourier4': np.stack(selected), 'fourier4_freephase': q8}
    for rank in [4, 8, 16, 72]:
        bases[f'pca{rank}'] = pca[:rank]
    for b in bases.values():
        assert np.max(np.abs(b.reshape(-1,24,4).mean(-1))) < 1e-10
        assert np.allclose(b @ b.T, np.eye(len(b)), atol=1e-9)
    return bases, labels, raw4


def oracle_l1(residual, basis):
    """Exact absolute-error projection; diagnostic only, uses held-out targets."""
    rank, n = basis.shape
    a = np.block([[basis.T, -np.eye(n)], [-basis.T, -np.eye(n)]])
    cost = np.r_[np.zeros(rank), np.ones(n)]
    bounds = [(None, None)]*rank + [(0, None)]*n
    pred = []
    for row in residual:
        fit = linprog(cost, A_ub=a, b_ub=np.r_[row, -row], bounds=bounds, method='highs')
        assert fit.success, fit.message
        p = fit.x[:rank] @ basis
        assert np.abs(p-row).mean() <= np.abs(row).mean()+1e-7
        pred.append(p)
    return np.stack(pred)


def score(pred, truth, hours):
    assert pred.shape == truth.shape and np.isfinite(pred).all()
    err = float(np.abs(pred.reshape(-1,4,24,4).mean(-1)-hours).max())
    assert err < 1e-8
    ae = np.abs(pred-truth).reshape(-1,4,96)
    return dict(mae=float(ae.mean()), by_zone=ae.mean((0,2)).tolist(),
                daily_mae=ae.mean((1,2)).tolist(), max_hourly_mean_error=err)


def train_model(x, residual, basis, family, lr, seed, epochs):
    import torch
    from torch import nn
    torch.set_num_threads(1); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xt = torch.tensor(x, dtype=torch.float32, device='cuda')
    rt = torch.tensor(residual/20, dtype=torch.float32, device='cuda')
    bt = torch.tensor(basis, dtype=torch.float32, device='cuda')
    net = (nn.Linear(x.shape[1], len(basis)) if family == 'linear' else
           nn.Sequential(nn.Linear(x.shape[1],64), nn.SiLU(), nn.Linear(64,len(basis)))).cuda()
    last = net if family == 'linear' else net[-1]
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    best, state, best_epoch = float('inf'), None, 0
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(302*4,device='cuda').split(128):
            loss = (net(xt[ids]) @ bt - rt[ids]).abs().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5); opt.step()
        net.eval()
        with torch.no_grad():
            val = float((net(xt[1208:1332]) @ bt-rt[1208:1332]).abs().mean())
        assert np.isfinite(val)
        if val < best:
            best, best_epoch = val, epoch+1
            state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch >= 30:
            break
    net.load_state_dict(state)
    with torch.no_grad():
        coefficients = net(xt[1208:]).cpu().numpy().astype('float64')*20
    parameters = sum(p.numel() for p in net.parameters())
    return coefficients @ basis, dict(family=family, lr=lr, seed=seed, best_epoch=best_epoch,
           epochs_run=epoch+1, parameters=parameters, validation_mae=best*20), state, coefficients


def main():
    import torch
    from reconstruction import read_targets, baseline, rich_features
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    out = OUT/'smoke' if args.smoke else OUT
    out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    source=Path('train_prices_15min.csv')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates, yy=read_targets(source); h=yy.mean(-1); anchor=baseline(h)
    assert len(dates)==363 and dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    # Recompute causal rich features, then give every zone its entire daily curve.
    f=rich_features(dates,h,dates,yy).reshape(363,4,24,-1)
    changed=yy.copy(); changed[333:]+=np.arange(4)*1234
    probe=rich_features(dates[333:334],h[333:334],dates,changed).reshape(1,4,24,-1)
    assert np.array_equal(f[333:334],probe)
    xx=f.reshape(363*4,-1)
    mu=xx[:1208].mean(0); sd=xx[:1208].std(0).clip(.05)
    x=np.c_[np.clip((xx-mu)/sd,-15,15),np.tile(np.eye(4),(363,1))].astype('float32')
    r=(yy-anchor).reshape(-1,96)
    bases, labels, raw4=make_bases(r[:1208])
    np.savez(out/'basis.npz',**bases, raw_fourier4=raw4, mu=mu,sd=sd)
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha,train_days=302,validation_days=31,
        test_days=30, fourier_modes=labels, output_budget='per day and zone; fourier4 and pca4 each four real weights',
        basis_fit='training only, uncentered residual SVD and greedy distinct-frequency SSE',
        features='causal rich features, all 24 hours flattened per zone; train-only scaling; zone one-hot',
        selection='August only; September developmental, previously viewed',
        oracle='exact L1 projection of held-out targets, diagnostic only; never predictor input'),indent=2))
    truth=yy[302:].reshape(-1,96); base=anchor[302:].reshape(-1,96)
    metrics={}; selections={}; trials=[]
    for name,basis in bases.items():
        if name=='pca72':
            assert np.max(np.abs((r[1208:] @ basis.T) @ basis-r[1208:])) < 1e-8
        rp=r[1208:1212] if args.smoke else r[1208:]
        oracle=oracle_l1(rp,basis)
        if args.smoke:
            assert np.isfinite(oracle).all(); print('ORACLE_SMOKE',name,flush=True)
        else:
            np.save(out/(name+'_oracle.npy'),base+oracle)
            metrics[name]={'oracle_august':score((base+oracle)[:124],truth[:124],h[302:333]),
                           'oracle_september':score((base+oracle)[124:],truth[124:],h[333:])}
            print('ORACLE',name,metrics[name]['oracle_september']['mae'],flush=True)
        # Rank72 is a capacity sanity check, not an extra prediction search.
        if name=='pca72': continue
        candidates=[]
        settings=[('linear',.003,17)] if args.smoke else [(f,lr,s) for f in ['linear','mlp'] for lr in [.001,.003] for s in [17,43]]
        for family,lr,seed in settings:
            pred,meta,state,coef=train_model(x,r,basis,family,lr,seed,3 if args.smoke else 300)
            folder=out/name/f'{family}_{lr}_{seed}'; folder.mkdir(parents=True,exist_ok=True)
            full=base+project(pred)
            meta.update(basis=name, outputs_per_day_zone=len(basis),
                        validation_mae=score(full[:124],truth[:124],h[302:333])['mae'])
            np.save(folder/'predictions.npy',full);np.save(folder/'coefficients.npy',coef)
            torch.save(dict(state=state,meta=meta),folder/'model.pt')
            # Replay saved weights in an identical architecture and compare coefficients.
            from torch import nn
            replay=(nn.Linear(x.shape[1],len(basis)) if family=='linear' else nn.Sequential(nn.Linear(x.shape[1],64),nn.SiLU(),nn.Linear(64,len(basis)))).cuda()
            replay.load_state_dict(torch.load(folder/'model.pt',weights_only=False)['state'])
            with torch.no_grad(): replay_coef=replay(torch.tensor(x[1208:],device='cuda')).cpu().numpy().astype('float64')*20
            assert np.array_equal(replay_coef,coef)
            (folder/'metrics.json').write_text(json.dumps(meta,indent=2))
            candidates.append((meta,full)); trials.append(meta)
            print('TRIAL',name,meta,flush=True)
        winner,pred=min(candidates,key=lambda c:c[0]['validation_mae'])
        selections[name]=winner
        np.save(out/(name+'_predictions.npy'),pred)
    if args.smoke:
        (out/'passed.json').write_text(json.dumps({'smoke':'passed','basis_labels':labels},indent=2)); return
    # Freeze selection before scoring any predictive September outputs.
    (out/'selection.json').write_text(json.dumps(selections,indent=2))
    for name,winner in selections.items():
        pred=np.load(out/(name+'_predictions.npy'))
        metrics[name].update(selected=winner,august=score(pred[:124],truth[:124],h[302:333]),
                             september=score(pred[124:],truth[124:],h[333:]))
        assert metrics[name]['oracle_september']['mae'] <= metrics[name]['september']['mae']+1e-6
    metrics['interpolation']={'august':score(base[:124],truth[:124],h[302:333]),'september':score(base[124:],truth[124:],h[333:])}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    lines=['# Four daily frequency weights — 2026-10-17','',
           'Fit: October–July. Selection: August. September is developmental, not blind.',
           'Oracle uses target-derived optimal weights and measures representation capacity only.', '',
           '| Basis | Daily weights / zone | August predictor MAE | September predictor MAE | September oracle MAE |',
           '|---|---:|---:|---:|---:|']
    for name,m in metrics.items():
        if name=='pca72':continue
        lines.append(f"| {name} | {len(bases[name]) if name in bases else 0} | {m['august']['mae']:.6f} | {m['september']['mae']:.6f} | {m.get('oracle_september',{}).get('mae',float('nan')):.6f} |")
    lines+=['','Fourier fixed-phase selections: '+json.dumps(labels),
            'Projected Fourier functions preserve hourly means; orthonormalization keeps their joint span.',
            'The four outputs are daily coefficients; the predictor has additional trainable parameters.',
            'Audits: source hash; causal first-September feature perturbation; train-only bases/scaling; finite outputs; hourly conservation; saved-weight replay; full-rank reconstruction; oracle dominates predictor.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('DONE',json.dumps({k:v.get('september',v.get('oracle_september'))['mae'] for k,v in metrics.items()}),flush=True)


if __name__=='__main__':
    main()
