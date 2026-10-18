"""Reproducible GPU search, chronological holdout, refit, offline inference QA."""
import argparse, hashlib, json, time, pickle, os
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from scipy.optimize import minimize
from reconstruction import read_targets, baseline, features, rich_features, Net, project

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--rich',action='store_true'); p.add_argument('--zone-ensemble',action='store_true'); p.add_argument('--out',default='results/search_gpu1'); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1', 'Explicit physical GPU1 required'
    dates,y=read_targets('train_prices_15min.csv'); h=y.mean(-1); b=baseline(h)
    feature_fn=rich_features if a.rich else features
    x=feature_fn(dates,h,dates,y); target=(y-b).reshape(-1,4)/20
    # Future/current targets must have no influence on an evaluated day's features.
    probe=len(dates)-60
    changed=y.copy(); changed[probe:]=changed[probe:]*17+1234
    assert np.array_equal(feature_fn(dates[probe:probe+1],h[probe:probe+1],dates,y),
                          feature_fn(dates[probe:probe+1],h[probe:probe+1],dates,changed))
    assert np.max(np.abs(b.mean(-1)-h))<1e-10
    # All zones/hours from a date stay together; last 60 dates excluded from selection.
    n=len(dates); tr=(n-120)*96; va=(n-60)*96
    mu=x[:tr].mean(0); sd=x[:tr].std(0).clip(.05)
    xx=np.clip((x-mu)/sd,-15,15)
    xt=torch.tensor(xx,device='cuda'); yt=torch.tensor(target,dtype=torch.float32,device='cuda')
    truth=y.reshape(-1,4); base=b.reshape(-1,4)
    candidates=[]; meta=[]; fitted=[]
    def add(name,pred,artifact,config):
        assert pred.shape==truth.shape and np.isfinite(pred).all()
        pred=project(pred-base)+base
        score=float(np.abs(pred[tr:va]-truth[tr:va]).mean())
        candidates.append(pred); fitted.append(artifact)
        meta.append(dict(name=name,validation_mae=score,config=config))
        (out/'validation_progress.json').write_text(json.dumps(meta,indent=2))
        print(name,'validation MAE',score,flush=True)
    add('interpolation',base,None,{})
    add('flat',np.repeat(h.reshape(-1,1),4,1),None,{})
    for alpha in ([100] if a.smoke else [1,10,100,1000,10000]):
        model=Ridge(alpha=alpha).fit(xx[:tr],target[:tr])
        add('ridge_'+str(alpha),base+20*project(model.predict(xx)),model,dict(alpha=alpha))
    for leaf in ([20] if a.smoke else [5,15,40,100]):
        cfg=dict(n_estimators=20 if a.smoke else 300,min_samples_leaf=leaf,max_features=.8,n_jobs=6,random_state=42)
        model=ExtraTreesRegressor(**cfg).fit(x[:tr],target[:tr])
        add('trees_'+str(leaf),base+20*project(model.predict(x)),model,cfg)
    if not a.smoke:
        for leaf,l2 in [(15,10),(31,10),(31,100),(63,100)]:
            cfg=dict(loss='absolute_error',max_iter=250,max_leaf_nodes=leaf,l2_regularization=l2,min_samples_leaf=40,learning_rate=.05,early_stopping=False,random_state=42)
            models=[HistGradientBoostingRegressor(**cfg).fit(x[:tr],target[:tr,q]) for q in range(4)]
            pp=np.column_stack([m.predict(x) for m in models])
            add('boost_'+str(leaf)+'_'+str(l2),base+20*project(pp),models,cfg)
    configs=[]
    for width,depth,drop in [(64,2,0),(128,2,.1),(256,2,.1),(128,3,.15)]:
        for lr in [.001,.0003]:
            for seed in [17,43]:
                configs.append(dict(width=width,depth=depth,dropout=drop,lr=lr,seed=seed))
    if a.smoke: configs=configs[:1]
    for cfg in configs:
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        model=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout']).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
        best=float('inf'); best_epoch=0; best_state=None
        for epoch in range(2 if a.smoke else 200):
            model.train()
            for inds in torch.randperm(tr,device='cuda').split(512):
                loss=(model(xt[inds])-yt[inds]).abs().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            model.eval()
            with torch.no_grad(): score=float((model(xt[tr:va])-yt[tr:va]).abs().mean())
            if score<best:
                best=score; best_epoch=epoch+1
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch-best_epoch>=25: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): pred=base+20*model(xt).cpu().numpy()
        config=dict(cfg,epochs=best_epoch)
        add('net_'+str(len(meta)),pred,best_state,config)
    # Select a sparse convex ensemble solely on the validation dates.
    preds=np.stack(candidates)
    errors=preds[:,tr:va]-truth[None,tr:va]
    m=len(meta); init=np.zeros(m); init[np.argmin([v['validation_mae'] for v in meta])]=1
    res=minimize(lambda w: np.abs(np.einsum('m,mnq->nq',w,errors)).mean(),init,
                 method='SLSQP',bounds=[(0,1)]*m,constraints={'type':'eq','fun':lambda w:w.sum()-1},
                 options={'maxiter':300,'ftol':1e-8})
    weights=res.x if res.success else init
    weights[weights<.005]=0; weights/=weights.sum()
    global_weights=weights.copy()
    zone_optimizer_success=[]
    if a.zone_ensemble:
        zone_weights=[]
        zonal_errors=errors.reshape(m,-1,4,24,4)
        for z in range(4):
            e=zonal_errors[:,:,z].reshape(m,-1)
            def objective(w):
                # Mild fixed shrinkage towards shared weights limits zone overfitting.
                return np.abs(w@e).mean()+.05*np.sum((w-global_weights)**2)
            zz=minimize(objective,global_weights,method='SLSQP',bounds=[(0,1)]*m,
                constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-8})
            ww=zz.x if zz.success else global_weights.copy()
            ww[ww<.005]=0; ww/=ww.sum()
            zone_weights.append(ww); zone_optimizer_success.append(bool(zz.success))
        weights=np.array(zone_weights).T
        chosen=(preds.reshape(m,-1,4,24,4)*weights[:,None,:,None,None]).sum(0).reshape(-1,4)
    else:
        chosen=np.einsum('m,mnq->nq',weights,preds)
    def scores(pred,sl):
        e=np.abs(pred[sl]-truth[sl]).reshape(-1,4,24,4)
        return dict(mae=float(e.mean()),mae_by_zone=e.mean((0,2,3)).tolist())
    report=dict(source_sha256=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest(),
        split=dict(train=[str(dates[0]),str(dates[n-121])],validation=[str(dates[n-120]),str(dates[n-61])],
                   holdout=[str(dates[n-60]),str(dates[-1])]),candidates=meta,weights=weights.tolist(),
        validation=scores(chosen,slice(tr,va)),holdout=scores(chosen,slice(va,None)),
        holdout_baseline=scores(base,slice(va,None)),holdout_flat=scores(candidates[1],slice(va,None)),
        max_hourly_mean_error=float(np.max(np.abs(chosen.mean(-1)-h.ravel()))),
        history_policy='Strictly earlier calendar dates; fixed 1,2,7-day lags and 7,28-day rolling means; no online fitting.',
        ensemble_optimizer_success=bool(res.success),zone_optimizer_success=zone_optimizer_success,
        ensemble_mode='zone' if a.zone_ensemble else 'global',global_weights=global_weights.tolist(),
        device=torch.cuda.get_device_name(0),smoke=a.smoke,feature_mode='rich' if a.rich else 'basic')
    assert report['max_hourly_mean_error']<1e-4
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'heldout_predictions.npz',dates=dates[-60:],observed=y[-60:],predicted=chosen[va:].reshape(-1,4,24,4),baseline=b[-60:])
    np.savez_compressed(out/'selection_predictions.npz',dates=dates,truth=y,baseline=b,candidates=preds)
    # Refit only selected candidates on all permitted data, with fixed selected epochs.
    bundle=dict(mu=mu,sd=sd,models=[],feature_count=x.shape[1],source_sha256=report['source_sha256'],feature_mode=report['feature_mode'])
    for i,w in enumerate(weights):
        if not np.any(w): continue
        name=meta[i]['name']; cfg=meta[i]['config']; artifact=None
        print('Refitting',name,'weight',w,flush=True)
        if name.startswith('ridge'): artifact=Ridge(**cfg).fit(xx,target)
        elif name.startswith('trees'): artifact=ExtraTreesRegressor(**cfg).fit(x,target)
        elif name.startswith('boost'): artifact=[HistGradientBoostingRegressor(**cfg).fit(x,target[:,q]) for q in range(4)]
        elif name.startswith('net'):
            torch.manual_seed(cfg['seed']); model=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout']).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
            for epoch in range(cfg['epochs']):
                model.train()
                for inds in torch.randperm(len(x),device='cuda').split(512):
                    loss=(model(xt[inds])-yt[inds]).abs().mean(); opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            artifact={k:v.detach().cpu() for k,v in model.state_dict().items()}
        bundle['models'].append(dict(name=name,weight=w.tolist() if a.zone_ensemble else float(w),config=cfg,artifact=artifact))
    with (out/'model.pkl').open('wb') as f: pickle.dump(bundle,f)
    # Verify the final frozen entry point from official-format files, including history isolation.
    from predict import predict
    last=dates[-1]; test=pd_hourly(last,h[-1]); test.to_csv(out/'qa_hourly.csv',index=False)
    import pandas as pd
    history=pd.read_csv('train_prices_15min.csv'); history[history.delivery_date<last].to_csv(out/'qa_history.csv',index=False)
    predict(out/'model.pkl',out/'qa_hourly.csv',out/'qa_history.csv',out/'qa_predictions.csv')
    got=pd.read_csv(out/'qa_predictions.csv'); assert len(got)==384
    keys=['delivery_date','market_id','hour_index']
    assert not got.duplicated(keys+['quarter_index']).any()
    assert np.isfinite(got.predicted_price_eur_mwh).all()
    grouped=got.groupby(keys).predicted_price_eur_mwh.agg(['mean','count'])
    check=grouped.join(test.set_index(keys))
    assert (check['count']==4).all()
    assert (check['mean']-check.hourly_price_eur_mwh).abs().max()<1e-4
    report['offline_inference_qa']='passed; 384 unique finite predictions and hourly means within tolerance'
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print('DONE',json.dumps({k:report[k] for k in ['validation','holdout','holdout_baseline','max_hourly_mean_error']}),flush=True)

def pd_hourly(date,h):
    import pandas as pd
    from reconstruction import ZONES
    return pd.DataFrame([(date,z,hr+1,h[zi,hr]) for zi,z in enumerate(ZONES) for hr in range(24)],columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh'])

if __name__=='__main__': main()
