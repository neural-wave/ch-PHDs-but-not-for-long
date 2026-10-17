"""Parallel, validation-selected simple-model search; September is diagnostic only."""
import argparse, concurrent.futures as cf, hashlib, itertools, json, os, time
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('SIMPLE_SEARCH_ROOT','results/simple_5090'))

def feature_modes():
    modes=os.environ.get('SIMPLE_SEARCH_FEATURES','compact,rich').split(',')
    assert modes and len(set(modes))==len(modes) and set(modes)<=set(['compact','rich','compact7'])
    return modes

def configs():
    rows=[]
    for feature,anchor in itertools.product(feature_modes(),['flat','interpolation']):
        for family in ['linear','additive','mlp','cnn']:
            shapes=([(0,0)] if family in ['linear','additive'] else
                    list(itertools.product([16,32,64,128],[1,2])) if family=='mlp' else
                    list(itertools.product([16,32],[3,5])))
            for (width,depth),lr,seed in itertools.product(shapes,[.001,.003], [17,43]):
                rows.append(dict(feature=feature,anchor=anchor,family=family,width=width,depth=depth,lr=lr,seed=seed))
    return [dict(id=i,**c) for i,c in enumerate(rows)]

def prepare():
    from reconstruction import read_targets,rich_features,baseline
    from interpretable_models import compact_features
    ROOT.mkdir(parents=True,exist_ok=True)
    source=Path('train_prices_15min.csv')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    assert len(dates)==363 and dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    changed=y.copy();changed[333:]+=np.arange(4)*1234
    raw={}
    raw['rich']=rich_features(dates,h,dates,y).reshape(363,4,24,-1)
    raw['compact']=compact_features(dates,h,dates,y,True)[0]
    if 'compact7' in feature_modes():
        raw['compact7']=compact_features(dates,h,dates,y,True,interpolation_days=7)[0]
    for name,x in raw.items():
        probe=(rich_features(dates[333:334],h[333:334],dates,changed).reshape(1,4,24,-1) if name=='rich' else
               compact_features(dates[333:334],h[333:334],dates,changed,True,interpolation_days=7 if name=='compact7' else 0)[0])
        assert np.array_equal(x[333:334],probe)
        mu=x[:302].mean((0,1,2));sd=x[:302].std((0,1,2)).clip(.05)
        xx=np.clip((x-mu)/sd,-15,15).astype('float32')
        if name in ['compact','compact7']:
            zone=np.broadcast_to(np.eye(4)[None,:,None,:],(363,4,24,4))
            xx=np.concatenate([xx,zone],-1).astype('float32')
        knots=np.quantile(xx[:302].reshape(-1,xx.shape[-1]),[.25,.5,.75],axis=0).astype('float32')
        np.savez(ROOT/(name+'.npz'),x=xx,mu=mu,sd=sd,knots=knots)
    np.savez(ROOT/'data.npz',dates=dates.astype('U10'),y=y,h=h,b=b)
    (ROOT/'manifest.json').write_text(json.dumps(dict(sha256=sha,train_days=302,validation_days=31,test_days=30,
        selection='August only; September developmental, never used for stopping or selection',candidates=configs(),
        feature_audit='current/future target perturbation leaves first September features unchanged'),indent=2))

def model(cfg,n,knots):
    import torch
    from torch import nn
    class Simple(nn.Module):
        def __init__(self):
            super().__init__();self.family=cfg['family']
            nin=n
            self.register_buffer('knots',torch.tensor(knots))
            if self.family in ['linear','additive']:
                self.net=nn.Linear(n*(4 if self.family=='additive' else 1),3)
            elif self.family=='mlp':
                layers=[]
                for _ in range(cfg['depth']):
                    layers += [nn.Linear(nin,cfg['width']),nn.SiLU()];nin=cfg['width']
                self.net=nn.Sequential(*layers,nn.Linear(nin,3))
            else:
                w=cfg['width'];k=cfg['depth']
                self.net=nn.Sequential(nn.Conv1d(n,w,k,padding=k//2,padding_mode='replicate'),nn.SiLU(),
                                       nn.Conv1d(w,w,k,padding=k//2,padding_mode='replicate'),nn.SiLU(),nn.Conv1d(w,3,1))
            last=self.net if isinstance(self.net,nn.Linear) else self.net[-1]
            nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
        def forward(self,x):
            if self.family=='additive':x=torch.cat([x]+[(x-k).relu() for k in self.knots],-1)
            if self.family=='cnn':
                b,z,t,n=x.shape;c=self.net(x.reshape(b*z,t,n).transpose(1,2)).transpose(1,2).reshape(b,z,t,3)
            else:c=self.net(x)
            a,u,v=c.unbind(-1)
            return torch.stack([a+u,a-u,-a+v,-a-v],-1)
    return Simple()

def trial(cfg,epochs=240,save=True):
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(cfg['seed']);start=time.perf_counter()
    d=np.load(ROOT/'data.npz');f=np.load(ROOT/(cfg['feature']+'.npz'))
    train_end=int(d['train_end']) if 'train_end' in d else 302
    select_end=int(d['select_end']) if 'select_end' in d else 333
    x=torch.tensor(f['x'],device='cuda');anchor=d['b'] if cfg['anchor']=='interpolation' else np.broadcast_to(d['h'][...,None],d['y'].shape)
    target=torch.tensor((d['y']-anchor)/20,dtype=torch.float32,device='cuda')
    net=model(cfg,x.shape[-1],f['knots']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
    best=float('inf');best_epoch=0;state=None
    for epoch in range(epochs):
        net.train()
        for ids in torch.randperm(train_end,device='cuda').split(32):
            loss=(net(x[ids])-target[ids]).abs().mean();opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        net.eval()
        with torch.no_grad():val=float((net(x[train_end:select_end])-target[train_end:select_end]).abs().mean())
        if not np.isfinite(val):raise RuntimeError('Nonfinite validation loss')
        if val<best:
            best=val;best_epoch=epoch+1;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch+1-best_epoch>=25:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():raw=net(x[train_end:]).cpu().numpy().astype('float64')
    pred=anchor[train_end:]+20*(raw-raw.mean(-1,keepdims=True))
    mean_error=float(np.abs(pred.mean(-1)-d['h'][train_end:]).max())
    assert np.isfinite(pred).all() and mean_error<1e-8
    meta=dict(config=cfg,parameters=sum(p.numel() for p in net.parameters()),validation_mae=float(np.abs(pred[:select_end-train_end]-d['y'][train_end:select_end]).mean()),
              best_epoch=best_epoch,epochs_run=epoch+1,seconds=time.perf_counter()-start,max_hourly_mean_error=mean_error)
    if save:
        folder=ROOT/'trials'/str(cfg['id']);folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(config=cfg,state=state,mu=f['mu'],sd=f['sd']),folder/'model.pt')
        np.save(folder/'predictions.npy',pred)
        loaded=torch.load(folder/'model.pt',weights_only=False,map_location='cpu')
        replay=model(cfg,x.shape[-1],f['knots']).cuda().eval();replay.load_state_dict(loaded['state'])
        with torch.no_grad():rr=replay(x[train_end:]).cpu().numpy().astype('float64')
        assert np.max(np.abs(rr-raw))<1e-7
        meta['replay_max_error']=float(np.max(np.abs(rr-raw)))
        (folder/'metrics.json').write_text(json.dumps(meta,indent=2))
    return meta

def pool_run(jobs,workers,epochs,save):
    import multiprocessing as mp
    results=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(trial,c,epochs,save) for c in jobs]
        for future in cf.as_completed(futures):
            result=future.result();results.append(result)
            if save:
                (ROOT/'progress.json').write_text(json.dumps(dict(completed=len(results),total=len(jobs),latest=result),indent=2))
                print('TRIAL',len(results),len(jobs),result,flush=True)
    return results

def finalize(results):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Save selection before touching September targets.
    selected=min(results,key=lambda r:r['validation_mae'])
    winners={family:min([r for r in results if r['config']['family']==family],key=lambda r:r['validation_mae']) for family in ['linear','additive','mlp','cnn']}
    (ROOT/'selection.json').write_text(json.dumps(dict(selected=selected,by_family=winners),indent=2))
    d=np.load(ROOT/'data.npz');truth=d['y'][333:];rows=[]
    for r in results:
        pred=np.load(ROOT/'trials'/str(r['config']['id'])/'predictions.npy')[31:]
        r.update(september_mae=float(np.abs(pred-truth).mean()),by_zone=np.abs(pred-truth).mean((0,2,3)).tolist())
        rows.append(dict(**r['config'],parameters=r['parameters'],validation_mae=r['validation_mae'],september_mae=r['september_mae'],seconds=r['seconds']))
    report=dict(selected=selected,by_family=winners,candidates=results,interpolation_september_mae=float(np.abs(d['b'][333:]-truth).mean()),
                audits='passed: source hash, causal feature perturbation, train-only normalization/knots, finite outputs, hourly conservation, all model serialized replay',
                caveat='September previously inspected; developmental test. August selects models/epochs. No ensemble or September fitting.')
    pd.DataFrame(rows).to_csv(ROOT/'scores.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for family,r in winners.items():ax.scatter(r['parameters'],r['september_mae'],s=65,label=family);ax.annotate(family,(r['parameters'],r['september_mae']),xytext=(5,5),textcoords='offset points')
    ax.axhline(5.234445,color='gray',ls='--',label='Prior single MLP (historical)');ax.set_xscale('log');ax.margins(x=.15,y=.15);ax.set(xlabel='Exact trainable parameters',ylabel='September MAE (EUR/MWh)',title='Simple models selected on August');ax.legend();fig.tight_layout()
    fig.savefig(ROOT/'simplicity.png',dpi=180);fig.savefig(ROOT/'simplicity.svg');plt.close(fig)
    (ROOT/'metrics.json').write_text(json.dumps(report,indent=2));print('DONE',selected,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.prepare:prepare();return
    if a.smoke:
        # Every family/feature/anchor exercised; benchmark same 16-job workload at 4/8/12 workers.
        jobs=[next(c for c in configs() if c['feature']==f and c['anchor']==b and c['family']==m) for f,b,m in itertools.product(feature_modes(),['flat','interpolation'],['linear','additive','mlp','cnn'])]
        profile=[]
        for workers in [4,8,12]:
            start=time.perf_counter();res=pool_run(jobs,workers,5,False)
            profile.append(dict(workers=workers,seconds=time.perf_counter()-start,jobs=len(res)))
            print('PROFILE',profile[-1],flush=True)
        (ROOT/'profile.json').write_text(json.dumps(profile,indent=2));return
    finalize(pool_run(configs(),a.workers,240,True))

if __name__=='__main__':main()
