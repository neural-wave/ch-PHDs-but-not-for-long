"""Bounded SSM study; all fitting on the authorized remote GPU."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,baseline
from interpretable_models import compact_features
from ssm_models import Model,S4D,S5
ROOT=Path('results/ssm_pilot_20261017')
SHA='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def configs():
    rows=[dict(family=f,width=w,state=32,lr=lr,seed=s) for f,w,lr,s in itertools.product(['s4d','s5','gru','tcn'],[16,32],[.001,.003],[17,43])]
    rows += [dict(family='compact',width=32,state=32,lr=lr,seed=s) for lr,s in itertools.product([.001,.003],[17,43])]
    return [dict(id=i,**r) for i,r in enumerate(rows)]


def features(dates,h,hd,hy):
    # Compact causal summary features are identical across all candidates.
    q=compact_features(dates,h,hd,hy,True)[0].transpose(0,2,1,3).reshape(len(dates),24,-1)
    hist={str(d):y for d,y in zip(hd,hy)};rows=[]
    for date in dates:
        chunks=[]
        for lag in range(7,0,-1):
            day=pd.Timestamp(str(date))-pd.Timedelta(days=lag);yy=hist.get(str(day.date()))
            z=np.zeros((24,4,10))
            if yy is not None:
                hh=yy.mean(-1);err=(yy-baseline(hh[None])[0])/20
                z[:,:,:4]=err.transpose(1,0,2);z[:,:,4:8]=np.abs(err).transpose(1,0,2)
                z[:,:,8]=1;z[:,:,9]=hh.T/100
            hour=np.arange(24);clock=np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.full(24,lag/7)],-1)
            chunks.append(np.concatenate([z.reshape(24,40),clock],-1))
        rows.append(np.concatenate(chunks,0))
    return q,np.asarray(rows)


def normalize(raw,fit_end,history=False):
    mu=raw[:fit_end].mean((0,1),keepdims=True);sd=raw[:fit_end].std((0,1),keepdims=True).clip(.05)
    x=np.clip((raw-mu)/sd,-15,15).astype('float32')
    if history:
        for zone in range(4):
            mask=raw[:,:,10*zone+8]
            x[:,:,10*zone:10*zone+10]*=mask[:,:,None]
            x[:,:,10*zone+8]=mask
    return x,mu,sd


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()==SHA
    dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);b=baseline(h)
    assert len(y)==363 and dates[302]=='2026-08-01' and dates[333]=='2026-09-01'
    q,r=features(dates,h,dates,y)
    # Whole target/future history unavailable; fixed known hourly curves.
    for idx in [120,302,333]:
        changed=y.copy();changed[idx:]+=123456
        qq,rr=features(dates[idx:idx+1],h[idx:idx+1],dates,changed)
        np.testing.assert_array_equal(q[idx:idx+1],qq);np.testing.assert_array_equal(r[idx:idx+1],rr)
    np.savez(ROOT/'data.npz',dates=dates.astype('U10'),y=y,h=h,b=b,q=q,r=r)
    write(ROOT/'manifest.json',dict(source_sha256=SHA,configs=configs(),fit_end=302,val_end=333,history='7 calendar days, 168 hourly tokens, all four zones jointly; missing days masked',selection='August only; September developmental',causal_perturbation=True))


def setup(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def smoke():
    setup(17);checks={}
    for cls in [S4D,S5]:
        layer=cls(8,16).cuda().double();x=torch.randn(2,168,8,device='cuda',dtype=torch.float64,requires_grad=True)
        a=layer(x);b=layer.reference(x);delta=float((a-b).abs().max().detach())
        assert delta<1e-8,(cls.__name__,delta)
        a.square().mean().backward();assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        assert float(layer.discrete()[0].abs().max().detach())<1
        checks[cls.__name__]=dict(recurrence_max_error=delta,finite_gradients=True,stable=True)
    d=np.load(ROOT/'data.npz');q=normalize(d['q'],302)[0];r=normalize(d['r'],302,True)[0]
    times={}
    for family in ['s4d','s5','gru','tcn','compact']:
        cfg=next(c for c in configs() if c['family']==family);net=Model(cfg,q.shape[-1],r.shape[-1]).cuda()
        opt=optimizer(net,cfg['lr']);xx=torch.tensor(q[:16],device='cuda');hh=torch.tensor(r[:16],device='cuda');target=torch.tensor((d['y'][:16]-d['b'][:16])/20,dtype=torch.float32,device='cuda')
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(3):
            out=net(xx,hh);loss=(out-target).abs().mean();opt.zero_grad();loss.backward();opt.step()
        torch.cuda.synchronize();times[family]=(time.perf_counter()-start)/3
        assert torch.isfinite(out).all() and float(out.sum(-1).abs().max().detach())<1e-5
    write(ROOT/'smoke.json',dict(cores=checks,batch16_step_seconds=times));print('SMOKE',checks,times,flush=True)


def optimizer(net,lr):
    state=[];normal=[]
    for name,p in net.named_parameters():
        (state if any(k in name for k in ['log_decay','freq','log_dt','.B']) else normal).append(p)
    return torch.optim.AdamW([dict(params=normal,lr=lr,weight_decay=.01),dict(params=state,lr=lr*.1,weight_decay=0)])


def train_trial(cfg,fit_end=302,val_end=333,stop=363,folder=None,epochs=160):
    setup(cfg['seed']);start=time.perf_counter();d=np.load(ROOT/'data.npz')
    q,muq,sdq=normalize(d['q'],fit_end);r,mur,sdr=normalize(d['r'],fit_end,True)
    q=torch.tensor(q,device='cuda');r=torch.tensor(r,device='cuda');target=torch.tensor((d['y']-d['b'])/20,dtype=torch.float32,device='cuda')
    net=Model(cfg,q.shape[-1],r.shape[-1]).cuda();opt=optimizer(net,cfg['lr']);best=float('inf');state=None;curves=[];be=0
    for epoch in range(1,epochs+1):
        net.train()
        for ids in torch.randperm(fit_end,device='cuda').split(32):
            loss=(net(q[ids],r[ids])-target[ids]).abs().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():val=float((net(q[fit_end:val_end],r[fit_end:val_end])-target[fit_end:val_end]).abs().mean())*20
        curves.append([epoch,val]);assert np.isfinite(val)
        if val<best:best=val;be=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-be>=20:break
    folder=folder or ROOT/'trials'/str(cfg['id']);folder.mkdir(parents=True,exist_ok=True)
    ck=dict(config=cfg,state=state,muq=muq,sdq=sdq,mur=mur,sdr=sdr,fit_end=fit_end,val_end=val_end,stop=stop)
    torch.save(ck,folder/'model.pt');np.savetxt(folder/'curves.csv',curves,delimiter=',',header='epoch,validation_mae')
    net.load_state_dict(state);net.eval()
    with torch.no_grad():raw=torch.cat([net(q[ids],r[ids]) for ids in torch.arange(fit_end,stop,device='cuda').split(32)]).cpu().numpy().astype('float64')
    pred=d['b'][fit_end:stop]+20*(raw-raw.mean(-1,keepdims=True));np.save(folder/'predictions.npy',pred)
    meta=dict(config=cfg,validation_mae=best,best_epoch=be,epochs_run=epoch,parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start,fit_end=fit_end,val_end=val_end,stop=stop)
    write(folder/'metrics.json',meta);return meta


def train():
    rows=[]
    for cfg in configs():
        folder=ROOT/'trials'/str(cfg['id'])
        row=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else train_trial(cfg)
        rows.append(row);write(ROOT/'progress.json',dict(completed=len(rows),total=len(configs()),latest=row));print('TRIAL',len(rows),row,flush=True)
    winners={f:min([r for r in rows if r['config']['family']==f],key=lambda r:r['validation_mae']) for f in ['s4d','s5','gru','tcn','compact']}
    write(ROOT/'selection.json',dict(candidates=rows,by_family=winners,overall=min(rows,key=lambda r:r['validation_mae'])))


def rolling():
    # Predetermined configurations: no reuse of August-selected hyperparameters.
    rows=[]
    for start,end in [(181,212),(212,243),(243,273),(273,302)]:
        for family in ['s4d','s5','gru','tcn','compact']:
            cfg=next(c for c in configs() if c['family']==family and c['width']==32 and c['lr']==.001 and c['seed']==17)
            folder=ROOT/'rolling'/f'{start}_{family}'
            row=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else train_trial(cfg,start-15,start,end,folder)
            rows.append(dict(folder=str(folder),**row));print('ROLLING',start,family,flush=True)
    write(ROOT/'rolling.json',dict(candidates=rows,note='Fixed preregistered width32/lr.001/seed17; 15-day preceding epoch selection; prior observed history allowed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','train','rolling']);a=p.parse_args();globals()[a.stage]()
