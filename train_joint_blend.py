"""Reweight existing selected leaves jointly; August only, no neural refitting."""
import hashlib,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
from scipy.optimize import minimize
from reconstruction import read_targets,baseline
from predict import bundle_prediction,predict

root=Path('results');out=root/'monthly_joint_blend_gpu1';out.mkdir(exist_ok=True)
selection=json.loads((root/'monthly_best.json').read_text());parent_dir=root/selection['best_directory']
report0=json.loads((parent_dir/'metrics.json').read_text())
assert report0['model_fit_days']==302 and report0['split_counts']==dict(train=302,validation=31,test=30)
with (parent_dir/'model.pkl').open('rb') as f:parent=pickle.load(f)
dates,y=read_targets('train_prices_15min.csv');truth=y.reshape(-1,4);h=y.mean(-1)
assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()==report0['source_sha256']
models=[];preds=[];meta=[]
def unpack(directory,bundle):
    report=json.loads((directory/'metrics.json').read_text())
    cache=np.load(directory/'selection_predictions.npz',allow_pickle=True)
    assert np.array_equal(cache['dates'],dates) and np.array_equal(cache['truth'],y)
    for item in bundle['models']:
        if item['name']=='parent_ensemble':
            unpack(root/report['parent'],item['artifact']);continue
        if item['name']=='interpolation' and any(m['name']=='interpolation' for m in models):continue
        idx=next(i for i,c in enumerate(report['candidates']) if c['name']==item['name'])
        models.append(dict(item));preds.append(cache['candidates'][idx])
        meta.append(dict(name=item['name'],config=item['config'],origin=directory.name))
unpack(parent_dir,parent)
# Retain the frozen parent as an explicit fallback.
cache=np.load(parent_dir/'selection_predictions.npz',allow_pickle=True)
w=np.asarray(report0['weights']);pp=(cache['candidates'].reshape(len(w),-1,4,24,4)*w[:,None,:,None,None]).sum(0).reshape(-1,4)
assert abs(np.abs(pp[302*96:333*96]-truth[302*96:333*96]).mean()-report0['validation']['mae'])<1e-8
models.append(dict(name='parent_ensemble',config={},artifact=parent));preds.append(pp);meta.append(dict(name='parent_ensemble',config={},origin=parent_dir.name))
preds=np.asarray(preds);m=len(preds);tr=302*96;va=333*96
err=preds[:,tr:va]-truth[None,tr:va]
init=np.zeros(m);init[-1]=1
rr=minimize(lambda w:np.abs(np.einsum('m,mnq->nq',w,err)).mean(),init,method='SLSQP',bounds=[(0,1)]*m,
    constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':500,'ftol':1e-9})
gw=rr.x if rr.success else init;gw[gw<.005]=0;gw/=gw.sum()
weights=[]
for z in range(4):
    ee=err.reshape(m,-1,4,24,4)[:,:,z].reshape(m,-1)
    rr=minimize(lambda w:np.abs(w@ee).mean()+.05*np.sum((w-gw)**2),gw,method='SLSQP',bounds=[(0,1)]*m,
        constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':500,'ftol':1e-9})
    ww=rr.x if rr.success else gw.copy();ww[ww<.005]=0;ww/=ww.sum();weights.append(ww)
weights=np.array(weights).T
blend=lambda w:(preds.reshape(m,-1,4,24,4)*w[:,None,:,None,None]).sum(0).reshape(-1,4)
alternatives=[np.repeat(gw[:,None],4,1),weights,np.repeat(init[:,None],4,1)]
weights=min(alternatives,key=lambda w:np.abs(blend(w)[tr:va]-truth[tr:va]).mean())
chosen=blend(weights)
bundle=dict(parent,models=[dict(item,weight=w.tolist()) for item,w in zip(models,weights) if np.any(w)])
with (out/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
score=lambda pred,sl:dict(mae=float(np.abs(pred[sl]-truth[sl]).mean()),mae_by_zone=np.abs(pred[sl]-truth[sl]).reshape(-1,4,24,4).mean((0,2,3)).tolist())
report=dict(report0,parent=parent_dir.name,candidates=meta,weights=weights.tolist(),validation=score(chosen,slice(tr,va)),test=score(chosen,slice(va,None)),
    method='Joint convex reweighting of existing selected components; August only; fixed 0.05 zone shrinkage; frozen parent fallback',
    max_hourly_mean_error=float(np.abs(chosen.mean(-1)-h.ravel()).max()))
assert report['max_hourly_mean_error']<1e-4
predict(out/'model.pkl',parent_dir/'qa_hourly.csv',parent_dir/'qa_history.csv',out/'qa_predictions.csv')
import pandas as pd
p=pd.read_csv(out/'qa_predictions.csv');keys=['delivery_date','market_id','hour_index'];assert len(p)==384 and not p.duplicated(keys+['quarter_index']).any()
ag=p.groupby(keys).predicted_price_eur_mwh.agg(['mean','count']).join(pd.read_csv(parent_dir/'qa_hourly.csv').set_index(keys))
assert np.isfinite(p.predicted_price_eur_mwh).all() and (ag['count']==4).all() and (ag['mean']-ag.hourly_price_eur_mwh).abs().max()<1e-4
report['offline_inference_qa']='passed; exported 384 unique finite rows and exact hourly means'
(out/'metrics.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(out/'selection_predictions.npz',dates=dates.astype('U10'),truth=y,baseline=baseline(h),candidates=preds)
np.savez_compressed(out/'test_predictions.npz',dates=dates[333:].astype('U10'),observed=y[333:],predicted=chosen[va:].reshape(-1,4,24,4),baseline=baseline(h)[333:])
print(json.dumps(dict(components=m,validation=report['validation'],test=report['test']),indent=2))
