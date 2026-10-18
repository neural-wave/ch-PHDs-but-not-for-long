"""Offline inference: --model MODEL --hourly test_hourly.csv --history history_15min.csv --output predictions.csv."""
import argparse, pickle
import numpy as np
import pandas as pd
import torch
from reconstruction import ZONES,read_targets,features,rich_features,baseline,project,Net
from catboost_models import cat_prediction,constrain_residual

def predict(model_path,hourly_path,history_path,output):
    with open(model_path,'rb') as f: bundle=pickle.load(f)
    df=pd.read_csv(hourly_path)
    assert not df.duplicated(['delivery_date','market_id','hour_index']).any()
    p=df.pivot(index='delivery_date',columns=['market_id','hour_index'],values='hourly_price_eur_mwh')
    p=p.reindex(columns=pd.MultiIndex.from_product([ZONES,range(1,25)])).sort_index()
    dates=p.index.to_numpy(); h=p.to_numpy().reshape(-1,4,24); assert np.isfinite(h).all()
    hd,hy=read_targets(history_path)
    assert all(str(d)<str(dates[0]) for d in hd), 'History must strictly precede evaluated dates'
    pred=bundle_prediction(bundle,dates,h,hd,hy)
    rows=[(d,z,hr+1,q+1,float(v)) for d,yy in zip(dates,pred.reshape(-1,4,24,4)) for z,zz in zip(ZONES,yy) for hr,hh in enumerate(zz) for q,v in enumerate(hh)]
    pd.DataFrame(rows,columns=['delivery_date','market_id','hour_index','quarter_index','predicted_price_eur_mwh']).to_csv(output,index=False,float_format='%.10f')

def bundle_prediction(bundle,dates,h,hd,hy):
    feature_fn=rich_features if bundle.get('feature_mode','basic')=='rich' else features
    x=feature_fn(dates,h,hd,hy); xx=np.clip((x-bundle['mu'])/bundle['sd'],-15,15)
    base=baseline(h).reshape(-1,4); pred=np.zeros_like(base)
    torch.set_num_threads(4)
    for item in bundle['models']:
        name=item['name']; model=item['artifact']; cfg=item['config']
        if name=='interpolation': pp=base
        elif name=='flat': pp=np.repeat(h.reshape(-1,1),4,1)
        elif name.startswith('ridge'): pp=base+20*project(model.predict(xx))
        elif name.startswith('trees'): pp=base+20*project(model.predict(x))
        elif name.startswith('boost'): pp=base+20*project(np.column_stack([m.predict(x) for m in model]))
        elif name.startswith('cat_'): pp=base+20*constrain_residual(cat_prediction(model,cfg,x),cfg)
        elif name=='parent_ensemble': pp=bundle_prediction(model,dates,h,hd,hy)
        else:
            net=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout']); net.load_state_dict(model); net.eval()
            with torch.no_grad(): pp=base+20*net(torch.tensor(xx)).numpy()
        weight=np.asarray(item['weight'])
        if weight.ndim:
            assert weight.shape==(4,)
            weight=np.tile(np.repeat(weight,24),len(dates))[:,None]
        pred+=weight*pp
    pred=project(pred)+h.reshape(-1,1)
    assert np.isfinite(pred).all() and np.max(np.abs(pred.mean(-1)-h.ravel()))<1e-4
    return pred

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--hourly',required=True); p.add_argument('--history',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    predict(a.model,a.hourly,a.history,a.output)
