"""Refit a frozen validation-selected pipeline on every official training day.
No settings or ensemble weights are selected during this final fit.
"""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'catboost_vendor'))
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from catboost import CatBoostRegressor
from reconstruction import read_targets,rich_features,baseline,Net
from graph_models import GraphNet
from revin_models import RevINGraph
from catboost_models import quarter_features
from predict import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    root=Path('results');selection=json.loads((root/'monthly_best.json').read_text());src=root/selection['best_directory']
    report=json.loads((src/'metrics.json').read_text())
    assert report['model_fit_days']==302 and report['split_counts']==dict(train=302,validation=31,test=30)
    assert report.get('offline_inference_qa','').startswith('passed')
    audit=json.loads((src/'audit.json').read_text());assert audit['status']=='passed'
    source_model_sha=hashlib.sha256((src/'model.pkl').read_bytes()).hexdigest();assert source_model_sha==audit['artifact_sha256']
    with (src/'model.pkl').open('rb') as f:original=pickle.load(f)
    dates,y=read_targets('train_prices_15min.csv');assert len(dates)==363
    source_sha=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest();assert source_sha==report['source_sha256']
    h=y.mean(-1);base=baseline(h);x=rich_features(dates,h,dates,y)
    mu=x.mean(0);sd=x.std(0).clip(.05);xx=np.clip((x-mu)/sd,-15,15);target=(y-base).reshape(-1,4)/20
    xt=torch.tensor(xx,device='cuda');yt=torch.tensor(target,dtype=torch.float32,device='cuda')
    ht=torch.tensor(h,dtype=torch.float32,device='cuda');xp=None
    out=root/('deployment_smoke_gpu1' if a.smoke else 'deployment_all363_gpu1');out.mkdir(exist_ok=True)
    fitted={};records=[];ntrain=len(dates)
    def refit_bundle(obj):
        nonlocal xp
        new=dict(obj,mu=mu,sd=sd,model_fit_days=ntrain,models=[])
        for item in obj['models']:
            name=item['name'];cfg=item['config'];old=item['artifact'];key=id(old)
            if name=='parent_ensemble':artifact=refit_bundle(old)
            elif name in ['flat','interpolation']:artifact=old
            elif key in fitted:artifact=fitted[key]
            else:
                print('REFIT',name,cfg,flush=True)
                if name.startswith('ridge'):artifact=clone(old).fit(xx,target)
                elif name.startswith('trees'):
                    artifact=clone(old)
                    if a.smoke:artifact.set_params(n_estimators=2)
                    artifact.fit(x,target)
                elif name.startswith('boost'):
                    artifact=[]
                    for q,m in enumerate(old):
                        mm=clone(m)
                        if a.smoke:mm.set_params(max_iter=2)
                        artifact.append(mm.fit(x,target[:,q]))
                elif name.startswith('cat_'):
                    def params(count):
                        return dict(iterations=2 if a.smoke else count,depth=cfg['depth'],learning_rate=cfg['learning_rate'],
                            l2_leaf_reg=cfg['l2_leaf_reg'],random_seed=cfg['seed'],loss_function='MAE',task_type='GPU',
                            devices='0',gpu_ram_part=.5,thread_count=4,border_count=128,allow_writing_files=False,verbose=False)
                    if cfg['strategy']=='pooled':
                        if xp is None:xp=quarter_features(x)
                        artifact=CatBoostRegressor(**params(cfg['epochs'][0])).fit(xp,target.ravel())
                    else:artifact=[CatBoostRegressor(**params(cfg['epochs'][q])).fit(x,target[:,q]) for q in range(4)]
                else:
                    torch.manual_seed(cfg['seed'])
                    joint=name.startswith('graph_') or name.startswith('revin_') or name.startswith('control_')
                    if name.startswith('graph_'):net=GraphNet(x.shape[1],cfg['width'],cfg['blocks'],cfg['dropout'])
                    elif name.startswith('revin_') or name.startswith('control_'):net=RevINGraph(x.shape[1],cfg['width'],cfg['blocks'],cfg['dropout'],cfg['revin'])
                    elif name.startswith('net'):net=Net(x.shape[1],cfg['width'],cfg['depth'],cfg['dropout'])
                    else:raise ValueError(name)
                    net=net.cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
                    for epoch in range(1 if a.smoke else cfg['epochs']):
                        net.train()
                        for inds in torch.randperm(ntrain if joint else len(x),device='cuda').split(16 if joint else 512):
                            if joint:
                                gx=xt.reshape(-1,4,24,x.shape[1])[inds]
                                pred=net(gx) if name.startswith('graph_') else net(gx,ht[inds])
                                loss=(pred-yt.reshape(-1,4,24,4)[inds]).abs().mean()
                            else:loss=(net(xt[inds])-yt[inds]).abs().mean()
                            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                    artifact={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    del net,opt
                fitted[key]=artifact;records.append(dict(name=name,config=cfg,fit_days=ntrain))
                (out/'progress.json').write_text(json.dumps(records,indent=2))
            new['models'].append(dict(item,artifact=artifact))
        return new
    bundle=refit_bundle(original)
    with (out/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
    # Eligibility smoke uses an official day, now included in fitting; it is not accuracy evaluation.
    qa_src=src
    while not (qa_src/'qa_hourly.csv').exists():qa_src=root/json.loads((qa_src/'metrics.json').read_text())['parent']
    predict(out/'model.pkl',qa_src/'qa_hourly.csv',qa_src/'qa_history.csv',out/'qa_predictions.csv')
    got=pd.read_csv(out/'qa_predictions.csv');inp=pd.read_csv(qa_src/'qa_hourly.csv');keys=['delivery_date','market_id','hour_index']
    assert len(got)==384 and not got.duplicated(keys+['quarter_index']).any() and np.isfinite(got.predicted_price_eur_mwh).all()
    agg=got.groupby(keys).predicted_price_eur_mwh.agg(['mean','count']).join(inp.set_index(keys));err=float((agg['mean']-agg.hourly_price_eur_mwh).abs().max())
    assert (agg['count']==4).all() and err<1e-4
    def check_weights(a,b):
        assert len(a['models'])==len(b['models'])
        for aa,bb in zip(a['models'],b['models']):
            assert aa['name']==bb['name'] and aa['config']==bb['config'] and aa['weight']==bb['weight']
            if aa['name']=='parent_ensemble':check_weights(aa['artifact'],bb['artifact'])
    check_weights(original,bundle)
    rr=dict(status='passed',smoke=a.smoke,model_fit_days=ntrain,normalization_days=ntrain,source_sha256=source_sha,
        selected_evaluation_pipeline=src.name,source_evaluation_model_sha256=source_model_sha,source_selection=selection,
        frozen_settings_and_weights=True,unique_refitted_components=records,tf32=False,
        accuracy_claim='No held-out accuracy claim for the all-363-day deployment refit; use the preserved 302-day evaluation model for August/September scores.',
        offline_inference_qa='passed; 384 finite unique exported rows and exact hourly means',max_exported_hourly_mean_error=err,
        artifact_sha256=hashlib.sha256((out/'model.pkl').read_bytes()).hexdigest())
    (out/'refit_report.json').write_text(json.dumps(rr,indent=2)+'\n')
    if not a.smoke:
        manifest=dict(selected_by='frozen_august_pipeline_refit_all_official_days',best_directory=out.name,model_fit_days=ntrain,
            evaluation_selection='monthly_best.json',source_evaluation_directory=src.name)
        (root/'deployment_best.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',json.dumps(dict(output=str(out),components=len(records),model_fit_days=ntrain,hourly_mean_error=err)),flush=True)
if __name__=='__main__':main()
