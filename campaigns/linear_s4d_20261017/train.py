"""S4D/S5 comparisons under VALIDATION_PROTOCOL.md; quantiles by default."""
import sys,hashlib,json,itertools,time,concurrent.futures as cf,multiprocessing as mp
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'campaigns/monthly_quantiles_20261017'))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from ssm_models import S4D
from ssm_experiment import optimizer,setup,SHA,write
from reconstruction import read_targets,baseline
import trainq as evaluation
ROOT=PROJECT/'results/linear_s4d_20261017';evaluation.ROOT=ROOT
TAUS=evaluation.TAUS
FAMILIES=['learned_frequencies','fixed_frequencies']


def configs():
    return [dict(id=i,family=f,width=10,state=n,lr=lr,seed=seed) for i,(f,n,lr,seed) in enumerate(itertools.product(FAMILIES,[8,16],[.001,.003],[17,43]))]


def features(dates,h,hd,hy):
    lookup={str(d):y for d,y in zip(hd,hy)};rows=[]
    for date,hh in zip(dates,h):
        chunks=[]
        for lag in range(7,0,-1):
            old=lookup.get(str((pd.Timestamp(str(date))-pd.Timedelta(days=lag)).date()))
            z=np.zeros((96,10))
            if old is not None:
                hourly=old.mean(-1);z[:,:4]=np.repeat(hourly,4,axis=-1).T/100
                z[:,4:8]=(old-baseline(hourly[None])[0]).reshape(4,96).T/20
                z[:,8:]=1
            chunks.append(z)
        query=np.zeros((96,10));query[:,:4]=np.repeat(hh,4,axis=-1).T/100;query[:,8]=1
        rows.append(np.concatenate(chunks+[query]))
    return np.zeros((len(dates),24,1)),np.asarray(rows)


def normalize(raw,nt,history=False):
    mu=raw[:nt].mean((0,1),keepdims=True);sd=raw[:nt].std((0,1),keepdims=True).clip(.05)
    # Preserve an affine point model: no clipping, nonlinear engineered features or norms.
    x=((raw-mu)/sd).astype('float32')
    if history:
        x[:,:,:4]*=raw[:,:,8:9];x[:,:,4:8]*=raw[:,:,9:10];x[:,:,8:]=raw[:,:,8:]
    return x,mu,sd


def folds(dates):
    months=np.asarray([str(d)[:7] for d in dates]);days=np.asarray([int(str(d)[8:10]) for d in dates]);out=[]
    for test in pd.period_range('2026-02','2026-09',freq='M'):
        val=str(test-1);test=str(test)
        out.append(dict(test=test,validation=val,nt=int((months<val).sum()),ns=int(((months==val)&(days<=15)).sum()),nc=int(((months==val)&(days>=16)).sum()),test_days=int((months==test).sum())))
    assert len(out)==8 and sum(f['test_days'] for f in out)==241
    return out


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);assert hashlib.sha256((PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==SHA
    dates,y=read_targets(PROJECT/'train_prices_15min.csv');h=y.mean(-1);b=baseline(h);q,r=features(dates,h,dates,y)
    for idx in [0,91,120,302,333]:
        changed=y.copy();changed[idx:]+=np.arange(4)*123456
        qq,rr=features(dates[idx:idx+1],h[idx:idx+1],dates,changed)
        np.testing.assert_array_equal(qq,q[idx:idx+1]);np.testing.assert_array_equal(rr,r[idx:idx+1])
    files=['ssm_models.py','ssm_experiment.py','reconstruction.py','interpretable_models.py','VALIDATION_PROTOCOL.md','campaigns/linear_s4d_20261017/train.py','campaigns/linear_s4d_20261017/audit.py','campaigns/linear_s4d_20261017/run.py','campaigns/linear_s4d_20261017/PROTOCOL.md','campaigns/linear_s4d_20261017/interpret.py','campaigns/monthly_quantiles_20261017/trainq.py']
    files += evaluation.p.DEPENDENCIES
    manifest=dict(source_sha256=SHA,folds=folds(dates),configs=configs(),taus=TAUS.tolist(),code_hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in files},causality_passed=True)
    if (ROOT/'manifest.json').exists():assert manifest==json.loads((ROOT/'manifest.json').read_text()),'Source changed; use new root'
    write(ROOT/'manifest.json',manifest);np.savez(ROOT/'raw.npz',dates=dates.astype('U10'),y=y,h=h,b=b,q=q,r=r)


class QuantileSSM(nn.Module):
    def __init__(self,cfg,qd,hd):
        super().__init__();self.cfg=cfg;assert hd==10
        self.core=S4D(10,cfg['state']);p=cfg['state']//2
        periods=np.array([1.,6.,12.,24.] if p==4 else [.5,1.,2.,4.,6.,12.,24.,168.])
        with torch.no_grad():
            self.core.log_dt.fill_(np.log(.25))  # fixed quarter-hour sampling; avoid A/dt confounding
            self.core.freq.copy_(torch.tensor(2*np.pi/periods,dtype=torch.float32)[None].expand(10,-1))
            self.core.log_decay.copy_(torch.tensor(np.log(1/(2*periods)),dtype=torch.float32)[None].expand(10,-1))
            self.core.C.mul_(.1)
        self.core.log_dt.requires_grad_(False)
        if cfg['family']=='fixed_frequencies':
            self.core.freq.requires_grad_(False)
        self.head=nn.Linear(20,44);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def initialize(self,target):
        e=np.quantile(target,TAUS,axis=(0,2,3)).T
        below=np.concatenate([np.maximum(-e[:,4:5],.005),np.maximum(np.diff(e[:,:5],axis=-1)[:,::-1],.005)],-1)
        above=np.concatenate([np.maximum(e[:,6:7],.005),np.maximum(np.diff(e[:,6:],axis=-1),.005)],-1)
        with torch.no_grad():self.head.bias[4:].copy_(torch.tensor(np.log(np.expm1(np.concatenate([below,above],-1))).ravel(),dtype=torch.float32))
    def representation(self,h):
        # One S4D instance shared in both directions over only permitted known inputs.
        return torch.cat([self.core(h),self.core(h.flip(1)).flip(1)],-1)[:,-96:]
    def forward(self,q,h):
        raw=self.head(self.representation(h));median=raw[...,:4].transpose(1,2).reshape(-1,4,24,4)
        median=median-median.mean(-1,keepdim=True)
        inc=F.softplus(raw[...,4:].reshape(-1,96,4,10).permute(0,2,1,3).reshape(-1,4,24,4,10))+1e-6
        return torch.cat([(median[...,None]-inc[...,:5].cumsum(-1)).flip(-1),median[...,None],median[...,None]+inc[...,5:].cumsum(-1)],-1)


def folder_data(fold,smoke=False):
    raw=np.load(ROOT/'raw.npz');n=sum(fold[k] for k in ['nt','ns','nc','test_days']);nt=fold['nt']
    folder=ROOT/('smoke' if smoke else 'folds')/fold['test'];folder.mkdir(parents=True,exist_ok=True)
    q,mq,sq=normalize(raw['q'][:n],nt);r,mr,sr=normalize(raw['r'][:n],nt,True)
    data={k:raw[k][:n] for k in ['dates','y','h','b']};data.update(q=q,r=r,muq=mq,sdq=sq,mur=mr,sdr=sr,nt=nt,ns=fold['ns'],nc=fold['nc'],selection_interpolation_mae=np.abs(raw['b'][nt:nt+fold['ns']]-raw['y'][nt:nt+fold['ns']]).mean())
    np.savez(folder/'data.npz',**data);write(folder/'manifest.json',fold);return folder


def fit(cfg,folder,smoke=False):
    setup(cfg['seed']);folder=Path(folder);dest=folder/'trials'/str(cfg['id']);start=time.perf_counter()
    d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns'])
    net=QuantileSSM(cfg,d['q'].shape[-1],d['r'].shape[-1]);net.initialize((d['y'][:nt]-d['b'][:nt])/20);net.cuda()
    q=torch.tensor(d['q'][:nt+ns],device='cuda');r=torch.tensor(d['r'][:nt+ns],device='cuda');target=torch.tensor((d['y'][:nt+ns]-d['b'][:nt+ns])/20,dtype=torch.float32,device='cuda');tau=torch.tensor(TAUS,dtype=torch.float32,device='cuda')
    opt=optimizer(net,cfg['lr']);best=float('inf');be=0;state=None;curves=[]
    for epoch in range(1,(2 if smoke else 300)+1):
        net.train()
        for ids in torch.randperm(nt,device='cuda').split(32):
            out=net(q[ids],r[ids]);err=target[ids,...,None]-out
            loss=err[...,5].abs().mean()+torch.maximum(tau*err,(tau-1)*err).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():out=net(q[nt:],r[nt:]);err=target[nt:,...,None]-out;val=float(err[...,5].abs().mean())*20;pb=float(torch.maximum(tau*err,(tau-1)*err).mean())*20
        curves.append([epoch,val/float(d['selection_interpolation_mae']),pb]);assert np.isfinite(val+pb)
        if val<best:best=val;be=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-be>=35:break
    net.load_state_dict(state);net.eval();out=[]
    with torch.no_grad():
        for first in range(nt,len(d['y']),32):out.append(net(torch.tensor(d['q'][first:first+32],device='cuda'),torch.tensor(d['r'][first:first+32],device='cuda')).cpu().numpy())
    quantiles=evaluation.physical(np.concatenate(out),d['b'][nt:]);dest.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=cfg,state=state),dest/'model.pt');np.save(dest/'quantiles.npy',quantiles)
    np.savetxt(dest/'curves.csv',curves,delimiter=',',header='epoch,selection_ratio,pinball',comments='')
    row=dict(config=cfg,validation_mae=float(np.abs(quantiles[:ns,...,5]-d['y'][nt:nt+ns]).mean()),selection_ratio=best/float(d['selection_interpolation_mae']),best_epoch=be,parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start)
    write(dest/'metrics.json',row);return row


def select(folder,rows):
    winners={f:min([r for r in rows if r['config']['family']==f],key=lambda r:(r['selection_ratio'],r['config']['id'])) for f in FAMILIES}
    winner=min(winners,key=lambda f:(winners[f]['selection_ratio'],f));winners['validation_selector']=dict(winners[winner],chosen_recipe=winner)
    d=np.load(folder/'data.npz');nt=int(d['nt']);emp=np.quantile(d['y'][:nt]-d['b'][:nt],TAUS,axis=(0,2)).transpose(1,2,0)
    emp[...,:5]=np.minimum(emp[...,:5],0);emp[...,5]=0;emp[...,6:]=np.maximum(emp[...,6:],0)
    dest=folder/'trials/interpolation';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',d['b'][nt:,...,None]+emp[None,:,None,:,:])
    winners['interpolation']=dict(config=dict(id='interpolation',family='interpolation'),selection_ratio=1.,validation_mae=float(d['selection_interpolation_mae']))
    write(folder/'selection.json',dict(by_recipe=winners,candidates=rows));return winners


def run(smoke=False):
    raw=np.load(ROOT/'raw.npz');jobs=folds(raw['dates']);reports=[]
    if smoke:jobs=[jobs[0],jobs[-1]]
    for fold in jobs:
        folder=folder_data(fold,smoke);candidates=configs()
        if smoke:candidates=[next(c for c in candidates if c['family']==f) for f in FAMILIES]
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for future in cf.as_completed([pool.submit(fit,c,str(folder),smoke) for c in candidates]):
                row=future.result();rows.append(row);write(folder/'progress.json',dict(completed=len(rows),total=len(candidates)));print('FIT',fold['test'],len(rows),row,flush=True)
        winners=select(folder,rows);report=evaluation.evaluate(folder,winners)
        d=np.load(folder/'data.npz');start=sum(int(d[k]) for k in ['nt','ns','nc']);truth=d['y'][start:];hh=d['h'][start:]
        uncorrected=np.empty_like(truth)
        for i in np.ndindex(hh.shape[:-1]):uncorrected[i]=np.interp(np.arange(96)/4+.125,np.arange(24)+.5,hh[i]).reshape(24,4)
        alternate=float(np.abs(uncorrected-truth).mean());zonebase=np.abs(d['b'][start:]-truth).mean((0,2,3))
        for score in report['results'].values():
            score['uncorrected_interpolation_mae']=alternate
            score['ratio_to_uncorrected_interpolation']=score['mae']/alternate if alternate>0 else None
            score['zone_interpolation_mae']=zonebase.tolist();score['zone_ratio']=(np.asarray(score['by_zone'])/zonebase).tolist()
        write(folder/'metrics.json',report);reports.append(report);evaluation.summarize(reports,smoke)
        basepool=np.average([v['results']['interpolation']['mae'] for v in reports],weights=[v['manifest']['test_days'] for v in reports])
        summarypath=(ROOT/'smoke' if smoke else ROOT)/'metrics.json';summary=json.loads(summarypath.read_text())
        for score in summary['summary']:score['pooled_interpolation_mae']=float(basepool);score['pooled_mae_ratio']=score['pooled_mae']/basepool
        write(summarypath,summary)
        write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_folds=len(reports),total_folds=len(jobs),last_test_month=fold['test']))
    if not smoke:pd.concat([pd.read_csv(ROOT/'folds'/f['test']/'predictions.csv.gz') for f in jobs]).to_csv(ROOT/'predictions.csv.gz',index=False,compression='gzip')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','run']);a=p.parse_args();prepare() if a.stage=='prepare' else run(a.stage=='smoke')
