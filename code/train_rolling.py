"""Expanding-prefix fits, 28-day validation, forward-only ensemble calibration."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from scipy.optimize import minimize
from reconstruction import ZONES,read_targets,rich_features,baseline,project,Net
from catboost_models import quarter_features,cat_prediction

def blend(preds,w):
    return (preds.reshape(len(preds),-1,4,24,4)*w[:,None,:,None,None]).sum(0).reshape(-1,4)

def weights_fit(preds,truth,prior,penalty):
    if penalty is None:return prior.copy()
    err=(preds-truth[None]).reshape(len(preds),-1,4,24,4);weights=[]
    for z in range(4):
        e=err[:,:,z].reshape(len(preds),-1);anchor=prior[:,z]
        result=minimize(lambda w:np.abs(w@e).mean()+penalty*np.sum((w-anchor)**2),anchor,
            method='SLSQP',bounds=[(0,1)]*len(preds),constraints={'type':'eq','fun':lambda w:w.sum()-1},
            options={'maxiter':250,'ftol':1e-8})
        w=result.x if result.success else anchor.copy()
        w=np.clip(w,0,1);w[w<1e-6]=0;w/=w.sum();weights.append(w)
    return np.asarray(weights).T

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='rolling_plan.json');p.add_argument('--smoke',action='store_true')
    p.add_argument('--out',default='results/rolling_search_gpu1');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    torch.set_num_threads(4);plan=json.loads(Path(a.plan).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    items=plan['components']
    if a.smoke:items=[v for v in items if v['name'] in ['interpolation','ridge_1000','net_15']]
    prior=np.asarray([v['anchor'] for v in items])
    empty=prior.sum(0)==0
    prior[:,empty]=1/len(items)
    prior/=prior.sum(0,keepdims=True)
    dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);b=baseline(h);truth=y.reshape(-1,4);base=b.reshape(-1,4)
    x=rich_features(dates,h,dates,y);target=(truth-base)/20
    assert len(dates)==363 and np.isfinite(x).all()
    assert np.max(np.abs(base.mean(-1)-h.ravel()))<1e-10
    # Retain fixed architecture/epochs from the prior search; no fold target early stopping.
    def fit(item,train_days,mu,sd):
        name=item['name'];cfg=item['config'];n=train_days*96
        xx=np.clip((x-mu)/sd,-15,15)
        if name in ['interpolation','flat']:return None
        if name.startswith('ridge'):return Ridge(**cfg).fit(xx[:n],target[:n])
        if name.startswith('trees'):
            cc=dict(cfg,n_estimators=10) if a.smoke else cfg
            return ExtraTreesRegressor(**cc).fit(x[:n],target[:n])
        if name.startswith('boost'):
            cc=dict(cfg,max_iter=3) if a.smoke else cfg
            return [HistGradientBoostingRegressor(**cc).fit(x[:n],target[:n,q]) for q in range(4)]
        if name.startswith('cat_'):
            from catboost import CatBoostRegressor
            def model(epochs):
                return CatBoostRegressor(iterations=3 if a.smoke else epochs,depth=cfg['depth'],
                    learning_rate=cfg['learning_rate'],l2_leaf_reg=cfg['l2_leaf_reg'],random_seed=cfg['seed'],
                    loss_function='MAE',task_type='GPU',devices='0',gpu_ram_part=.5,thread_count=4,
                    border_count=128,allow_writing_files=False,verbose=False)
            if cfg['strategy']=='pooled':
                obj=model(cfg['epochs'][0]);obj.fit(quarter_features(x[:n]),target[:n].ravel());return obj
            objects=[]
            for q in range(4):
                obj=model(cfg['epochs'][q]);obj.fit(x[:n],target[:n,q]);objects.append(obj)
            return objects
        torch.manual_seed(cfg['seed']);net=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout']).cuda()
        xt=torch.tensor(xx[:n],device='cuda');yt=torch.tensor(target[:n],dtype=torch.float32,device='cuda')
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
        for epoch in range(2 if a.smoke else cfg['epochs']):
            net.train()
            for inds in torch.randperm(n,device='cuda').split(512):
                loss=(net(xt[inds])-yt[inds]).abs().mean();opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    def predict_component(item,obj,sl,mu,sd):
        name=item['name'];cfg=item['config'];xx=np.clip((x[sl]-mu)/sd,-15,15)
        if name=='interpolation':return base[sl]
        if name=='flat':return np.repeat(h.reshape(-1,1)[sl],4,1)
        if name.startswith('ridge'):raw=obj.predict(xx)
        elif name.startswith('trees'):raw=obj.predict(x[sl])
        elif name.startswith('boost'):raw=np.column_stack([m.predict(x[sl]) for m in obj])
        elif name.startswith('cat_'):raw=cat_prediction(obj,cfg,x[sl])
        else:
            net=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout']);net.load_state_dict(obj);net.eval()
            with torch.no_grad():raw=net(torch.tensor(xx)).numpy()
        return base[sl]+20*project(np.asarray(raw,dtype=np.float64))
    folds=[];blocks=[];observed=[]
    for train_days in plan['fold_train_days']:
        end=train_days+plan['window_days'];n=train_days*96;sl=slice(n,end*96)
        mu=x[:n].mean(0);sd=x[:n].std(0).clip(.05);preds=[]
        for item in items:
            obj=fit(item,train_days,mu,sd);pred=predict_component(item,obj,sl,mu,sd)
            assert np.isfinite(pred).all();preds.append(pred)
            print('fold',train_days,item['name'],'mae',float(np.abs(pred-truth[sl]).mean()),flush=True)
        blocks.append(np.asarray(preds));observed.append(truth[sl])
        folds.append(dict(train_days=train_days,train_end=str(dates[train_days-1]),
            validation=[str(dates[train_days]),str(dates[end-1])],validation_days=end-train_days))
        (out/'progress.json').write_text(json.dumps(dict(completed_folds=folds),indent=2))
        np.savez_compressed(out/('fold_'+str(train_days)+'.npz'),candidates=preds,observed=truth[sl],dates=dates[train_days:end].astype('U10'))
    # Only EARLIER validation folds calibrate the weights used on a later fold.
    rules=[None]+plan['penalties'];forward=[]
    for penalty in rules:
        predictions=[];targets=[]
        for held in [1,2]:
            pp=np.concatenate(blocks[:held],axis=1);tt=np.concatenate(observed[:held])
            w=weights_fit(pp,tt,prior,penalty)
            predictions.append(blend(blocks[held],w));targets.append(observed[held])
        score=float(np.abs(np.concatenate(predictions)-np.concatenate(targets)).mean())
        forward.append(dict(penalty=penalty,mae=score))
    best=min(forward,key=lambda r:r['mae']);all_preds=np.concatenate(blocks,axis=1);all_truth=np.concatenate(observed)
    chosen_weights=weights_fit(all_preds,all_truth,prior,best['penalty'])
    print('FORWARD RULES',forward,'selected',best,flush=True)
    # The audit is scored only after the rule is frozen, and never chooses the rule.
    audit_start=plan['audit_train_days'];n=audit_start*96;sl=slice(n,None)
    mu=x[:n].mean(0);sd=x[:n].std(0).clip(.05);audit_preds=[]
    for item in items:
        obj=fit(item,audit_start,mu,sd);audit_preds.append(predict_component(item,obj,sl,mu,sd))
    audit_preds=np.asarray(audit_preds);chosen=blend(audit_preds,chosen_weights);anchor=blend(audit_preds,prior)
    def score(pred,tt):
        e=np.abs(pred-tt).reshape(-1,4,24,4)
        return dict(mae=float(e.mean()),mae_by_zone=e.mean((0,2,3)).tolist())
    report=dict(source_sha256=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest(),
        protocol='forward_28day_cv_mae',folds=folds,forward_rules=forward,selected_rule=best,
        forward_validation_days=56,total_oof_days=84,window_days=28,audit_days=len(dates)-audit_start,
        audit_dates=[str(dates[audit_start]),str(dates[-1])],audit=score(chosen,truth[sl]),
        audit_anchor_same_fits=score(anchor,truth[sl]),audit_interpolation=score(base[sl],truth[sl]),
        max_hourly_mean_error=float(np.abs(chosen.mean(-1)-h[audit_start:].ravel()).max()),
        weights=chosen_weights.tolist(),prior_weights=prior.tolist(),components=items,smoke=a.smoke,
        caveat=plan['caveat'],device=torch.cuda.get_device_name(0))
    assert report['max_hourly_mean_error']<1e-4
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'audit_predictions.npz',dates=dates[audit_start:].astype('U10'),observed=y[audit_start:],
        predicted=chosen.reshape(-1,4,24,4),anchor=anchor.reshape(-1,4,24,4),baseline=b[audit_start:])
    mu=x.mean(0);sd=x.std(0).clip(.05)
    bundle=dict(mu=mu,sd=sd,feature_count=x.shape[1],feature_mode='rich',source_sha256=report['source_sha256'],models=[])
    for item,w in zip(items,chosen_weights):
        if not np.any(w):continue
        print('Final refit',item['name'],w,flush=True)
        obj=fit(item,len(dates),mu,sd)
        bundle['models'].append(dict(name=item['name'],config=item['config'],artifact=obj,weight=w.tolist()))
    with (out/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
    from predict import predict
    pd.DataFrame([(dates[-1],zone,hr+1,h[-1,z,hr]) for z,zone in enumerate(ZONES) for hr in range(24)],
        columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh']).to_csv(out/'qa_hourly.csv',index=False)
    history=pd.read_csv('train_prices_15min.csv');history[history.delivery_date<dates[-1]].to_csv(out/'qa_history.csv',index=False)
    predict(out/'model.pkl',out/'qa_hourly.csv',out/'qa_history.csv',out/'qa_predictions.csv')
    got=pd.read_csv(out/'qa_predictions.csv');keys=['delivery_date','market_id','hour_index']
    assert len(got)==384 and not got.duplicated(keys+['quarter_index']).any() and np.isfinite(got.predicted_price_eur_mwh).all()
    agg=got.groupby(keys).predicted_price_eur_mwh.agg(['mean','count']).join(pd.read_csv(out/'qa_hourly.csv').set_index(keys))
    assert (agg['count']==4).all() and (agg['mean']-agg.hourly_price_eur_mwh).abs().max()<1e-4
    report['offline_inference_qa']='passed; 384 unique finite predictions and hourly means within tolerance'
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    if not a.smoke:
        Path('results/rolling_best.json').write_text(json.dumps(dict(selected_by='forward_28day_cv_mae',
            best_directory=out.name,selected_rule=best),indent=2))
    print('DONE',json.dumps({k:report[k] for k in ['selected_rule','audit','audit_anchor_same_fits','audit_interpolation']}),flush=True)

if __name__=='__main__':main()
