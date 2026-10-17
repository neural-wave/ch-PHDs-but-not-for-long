"""Independent saved-artifact provenance, frozen-settings and inference audit."""
import hashlib,json,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,rich_features,ZONES
from predict import predict,bundle_prediction
root=Path('results');directory=root/'deployment_all363_gpu1';r=json.loads((directory/'refit_report.json').read_text())
assert r['status']=='passed' and not r['smoke'] and r['model_fit_days']==363 and r['frozen_settings_and_weights']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(directory/'model.pkl')==r['artifact_sha256']
assert sha('train_prices_15min.csv')==r['source_sha256']
with (directory/'model.pkl').open('rb') as f:bundle=pickle.load(f)
with (root/r['selected_evaluation_pipeline']/'model.pkl').open('rb') as f:old=pickle.load(f)
dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);x=rich_features(dates,h,dates,y)
mu=x.mean(0);sd=x.std(0).clip(.05);seen=set();counts={};levels=0
leaves={}
def inspect(a,b,ancestor):
    global levels
    levels+=1
    assert a['model_fit_days']==363 and a['source_sha256']==r['source_sha256']
    assert np.array_equal(a['mu'],mu) and np.array_equal(a['sd'],sd)
    assert len(a['models'])==len(b['models'])
    total=np.asarray([i['weight'] for i in a['models']]).sum(0);assert np.allclose(total,1,atol=1e-10)
    for aa,bb in zip(a['models'],b['models']):
        name=aa['name'];artifact=aa['artifact'];cfg=aa['config'];w=ancestor*np.asarray(aa['weight'])
        assert name==bb['name'] and cfg==bb['config'] and aa['weight']==bb['weight']
        if name=='parent_ensemble':inspect(artifact,bb['artifact'],w);continue
        key=(name,json.dumps(cfg,sort_keys=True))
        if key not in leaves:leaves[key]=dict(aa,weight=np.zeros(4))
        leaves[key]['weight']+=w
        if name in ['flat','interpolation'] or id(artifact) in seen:continue
        seen.add(id(artifact));kind=name.split('_')[0];counts[kind]=counts.get(kind,0)+1
        if kind=='trees':assert all(e.tree_.n_node_samples[0]==363*96 for e in artifact.estimators_) and len(artifact.estimators_)==cfg['n_estimators']
        elif kind=='boost':assert all(m.n_iter_==cfg['max_iter'] for m in artifact)
        elif kind=='cat':assert [m.tree_count_ for m in artifact]==cfg['epochs']
        elif kind=='ridge':assert np.isfinite(artifact.coef_).all()
        else:assert all(torch.isfinite(v).all() for v in artifact.values())
inspect(bundle,old,np.ones(4));assert len(seen)==len(r['unique_refitted_components'])==22
recipe=json.loads(Path('frozen_recipe.json').read_text())
assert len(recipe['models'])==len(leaves)
for item in recipe['models']:
    leaf=leaves[(item['name'],json.dumps(item['config'],sort_keys=True))]
    assert np.allclose(item['weight'],leaf['weight'],rtol=0,atol=1e-12)
flat=dict(bundle,models=[dict(i,weight=i['weight'].tolist()) for i in leaves.values()])
qa=directory/'audit_inputs';qa.mkdir(exist_ok=True);cases=[]
for label,hh,hd,hy in [('official_last_day',h[-1:],dates[:-1],y[:-1]),('negative_flat_empty_history',np.full((1,4,24),-25.),dates[:0],y[:0]),('volatile_supplied_curve',h[-1:]*2,dates[:-1],y[:-1])]:
    dd=dates[-1:]
    pred=bundle_prediction(bundle,dd,hh,hd,hy);fp=bundle_prediction(flat,dd,hh,hd,hy)
    difference=float(np.abs(pred-fp).max());assert difference<1e-7
    hourly=qa/(label+'_hourly.csv');history=qa/(label+'_history.csv');output=qa/(label+'_predictions.csv')
    pd.DataFrame([(dd[0],z,hr+1,hh[0,zi,hr]) for zi,z in enumerate(ZONES) for hr in range(24)],columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh']).sample(frac=1,random_state=17).to_csv(hourly,index=False)
    pd.DataFrame([(d,z,hr+1,q+1,yy[zi,hr,q]) for d,yy in zip(hd,hy) for zi,z in enumerate(ZONES) for hr in range(24) for q in range(4)],columns=['delivery_date','market_id','hour_index','quarter_index','price_eur_mwh']).to_csv(history,index=False)
    predict(directory/'model.pkl',hourly,history,output)
    got=pd.read_csv(output);keys=['delivery_date','market_id','hour_index'];assert len(got)==384 and np.isfinite(got.predicted_price_eur_mwh).all() and not got.duplicated(keys+['quarter_index']).any()
    ag=got.groupby(keys).predicted_price_eur_mwh.agg(['mean','count']).join(pd.read_csv(hourly).set_index(keys));err=float((ag['mean']-ag.hourly_price_eur_mwh).abs().max());assert (ag['count']==4).all() and err<1e-4
    cases.append(dict(case=label,rows=384,hourly_mean_error=err,flattened_max_difference=difference))
with (directory/'model_flat.pkl').open('wb') as f:pickle.dump(flat,f)
result=dict(status='passed',model_fit_days=363,normalization_days=363,bundle_levels=levels,unique_components=counts,
    frozen_settings_weights_and_recipe_match=True,artifact_sha256=r['artifact_sha256'],flat_artifact_sha256=sha(directory/'model_flat.pkl'),
    synthetic_eligibility_cases=cases,accuracy_claim=r['accuracy_claim'])
(directory/'deployment_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
