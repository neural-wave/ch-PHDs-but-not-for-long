"""Joint-node model search with August validation and September development test."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from reconstruction import ZONES,read_targets,rich_features,baseline,project
from graph_models import GraphNet

def blend(preds,w):
    return (preds.reshape(len(preds),-1,4,24,4)*w[:,None,:,None,None]).sum(0).reshape(-1,4)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--out',default='results/monthly_graph_gpu1');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path('results/monthly_best.json').read_text());parent_dir=Path('results')/selection['best_directory']
    assert selection['selected_by']=='august_validation_mae' and parent_dir.name in ['monthly_base_gpu1','monthly_cat_gpu1']
    report0=json.loads((parent_dir/'metrics.json').read_text())
    assert report0['split_counts']==dict(train=302,validation=31,test=30)
    assert report0['model_fit_days']==302 and report0['final_refit_deferred']
    assert report0.get('offline_inference_qa','').startswith('passed')
    dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);b=baseline(h);tr=302*96;va=333*96
    sha=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest();assert sha==report0['source_sha256']
    cache=np.load(parent_dir/'selection_predictions.npz',allow_pickle=True)
    assert np.array_equal(cache['dates'],dates) and np.array_equal(cache['truth'],y)
    cp=cache['candidates'];ww=np.asarray(report0['weights'])
    parent_prediction=blend(cp,ww) if ww.ndim==2 else np.einsum('m,mnq->nq',ww,cp)
    truth=y.reshape(-1,4);base=b.reshape(-1,4)
    assert abs(np.abs(parent_prediction[tr:va]-truth[tr:va]).mean()-report0['validation']['mae'])<1e-8
    x=rich_features(dates,h,dates,y)
    with (parent_dir/'model.pkl').open('rb') as f:parent=pickle.load(f)
    mu=parent['mu'];sd=parent['sd'];xx=np.clip((x-mu)/sd,-15,15).reshape(-1,4,24,x.shape[1])
    xt=torch.tensor(xx,device='cuda');yt=torch.tensor((y-b)/20,dtype=torch.float32,device='cuda')
    predictions=[parent_prediction,base];states=[parent,None];meta=[dict(name='parent_ensemble',config={}),dict(name='interpolation',config={})]
    configs=[]
    for width,blocks,dropout in [(32,1,.1),(64,1,.1),(64,2,.15),(128,2,.2)]:
        for lr in [.001,.0003]:
            for seed in [17,43]:configs.append(dict(width=width,blocks=blocks,dropout=dropout,lr=lr,seed=seed))
    if a.smoke:configs=configs[:1]
    for i,cfg in enumerate(configs):
        torch.manual_seed(cfg['seed']);net=GraphNet(x.shape[1],cfg['width'],cfg['blocks'],cfg['dropout']).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
        best=float('inf');best_epoch=0;state=None
        for epoch in range(2 if a.smoke else 250):
            net.train()
            for inds in torch.randperm(302,device='cuda').split(16):
                loss=(net(xt[inds])-yt[inds]).abs().mean();opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            net.eval()
            with torch.no_grad():score=float((net(xt[302:333])-yt[302:333]).abs().mean())
            if score<best:
                best=score;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            if epoch-best_epoch>=30:break
        net.load_state_dict(state);net.eval()
        with torch.no_grad():raw=net(xt).cpu().numpy().reshape(-1,4)
        pred=base+20*project(np.asarray(raw,dtype=np.float64))
        assert np.isfinite(pred).all()
        cfg=dict(cfg,epochs=best_epoch)
        meta.append(dict(name='graph_'+str(i),config=cfg,validation_mae=float(np.abs(pred[tr:va]-truth[tr:va]).mean())))
        states.append(state);predictions.append(pred)
        (out/'validation_progress.json').write_text(json.dumps(meta,indent=2))
        print(meta[-1]['name'],'August MAE',meta[-1]['validation_mae'],'epoch',best_epoch,flush=True)
    preds=np.asarray(predictions);err=preds[:,tr:va]-truth[None,tr:va];m=len(preds)
    init=np.zeros(m);init[np.argmin(np.abs(err).mean((1,2)))]=1
    result=minimize(lambda w:np.abs(np.einsum('m,mnq->nq',w,err)).mean(),init,method='SLSQP',bounds=[(0,1)]*m,
        constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-8})
    global_weights=result.x if result.success else init;global_weights[global_weights<.005]=0;global_weights/=global_weights.sum()
    columns=[]
    for z in range(4):
        ee=err.reshape(m,-1,4,24,4)[:,:,z].reshape(m,-1)
        rr=minimize(lambda w:np.abs(w@ee).mean()+.05*np.sum((w-global_weights)**2),global_weights,method='SLSQP',
            bounds=[(0,1)]*m,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-8})
        w=rr.x if rr.success else global_weights.copy();w[w<.005]=0;w/=w.sum();columns.append(w)
    weights=np.asarray(columns).T
    # Choose the lower-August-scoring mixture; never use September for this decision.
    global_matrix=np.repeat(global_weights[:,None],4,axis=1)
    if np.abs(blend(preds,global_matrix)[tr:va]-truth[tr:va]).mean()<np.abs(blend(preds,weights)[tr:va]-truth[tr:va]).mean():weights=global_matrix
    chosen=blend(preds,weights)
    def score(pp,sl):
        ee=np.abs(pp[sl]-truth[sl]).reshape(-1,4,24,4)
        return dict(mae=float(ee.mean()),mae_by_zone=ee.mean((0,2,3)).tolist())
    report=dict(source_sha256=sha,split=report0['split'],split_counts=report0['split_counts'],parent=parent_dir.name,
        calendar_split=True,model_fit_days=302,final_refit_deferred=True,graph_nodes=4,graph_hours=24,
        candidates=meta,weights=weights.tolist(),validation=score(chosen,slice(tr,va)),test=score(chosen,slice(va,None)),
        test_baseline=score(base,slice(va,None)),max_hourly_mean_error=float(np.abs(chosen.mean(-1)-h.ravel()).max()),
        smoke=a.smoke,device=torch.cuda.get_device_name(0))
    assert report['max_hourly_mean_error']<1e-4
    bundle=dict(mu=mu,sd=sd,feature_mode='rich',feature_count=x.shape[1],source_sha256=sha,models=[])
    for item,state,w in zip(meta,states,weights):
        if np.any(w):bundle['models'].append(dict(name=item['name'],config=item['config'],weight=w.tolist(),artifact=state))
    with (out/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
    np.savez_compressed(out/'test_predictions.npz',dates=dates[333:].astype('U10'),observed=y[333:],predicted=chosen[va:].reshape(-1,4,24,4),baseline=b[333:])
    np.savez_compressed(out/'selection_predictions.npz',dates=dates.astype('U10'),truth=y,baseline=b,candidates=preds)
    # Exercise graph inference even when the short smoke ensemble chooses only its parent.
    graph_bundle=dict(bundle,models=[dict(name=meta[2]['name'],config=meta[2]['config'],weight=[1]*4,artifact=states[2])])
    with (out/'graph_only_qa.pkl').open('wb') as f:pickle.dump(graph_bundle,f)
    from predict import predict
    pd.DataFrame([(dates[-1],zone,hr+1,h[-1,z,hr]) for z,zone in enumerate(ZONES) for hr in range(24)],
        columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh']).to_csv(out/'qa_hourly.csv',index=False)
    history=pd.read_csv('train_prices_15min.csv');history[history.delivery_date<dates[-1]].to_csv(out/'qa_history.csv',index=False)
    for model_name,csv_name in [('model.pkl','qa_predictions.csv'),('graph_only_qa.pkl','graph_only_qa.csv')]:
        predict(out/model_name,out/'qa_hourly.csv',out/'qa_history.csv',out/csv_name)
        got=pd.read_csv(out/csv_name);keys=['delivery_date','market_id','hour_index']
        assert len(got)==384 and not got.duplicated(keys+['quarter_index']).any() and np.isfinite(got.predicted_price_eur_mwh).all()
        agg=got.groupby(keys).predicted_price_eur_mwh.agg(['mean','count']).join(pd.read_csv(out/'qa_hourly.csv').set_index(keys))
        assert (agg['count']==4).all() and (agg['mean']-agg.hourly_price_eur_mwh).abs().max()<1e-4
    report['offline_inference_qa']='passed; ensemble and graph-only exported CSVs preserve finite unique four-quarter hourly means'
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print('DONE August',report['validation'],'September',report['test'],flush=True)

if __name__=='__main__':main()
