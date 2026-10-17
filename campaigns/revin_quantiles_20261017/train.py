"""Matched fixed, fitting-z-score and reversible daily-normalization comparison."""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('ssm_reference',PROJECT/'campaigns/ssm_monthly_20261017/train.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,baseline

ROOT=PROJECT/'results/revin_quantiles_20261017'
MODES=['fixed','day_reversible']
FAMILIES=['compact']
RECIPES=[f'{f}_{m}' for f,m in itertools.product(FAMILIES,MODES)]
TAUS=old.TAUS
evaluation=old.evaluation
evaluation.ROOT=ROOT
setup=old.setup
write=old.write
SHA=old.SHA
folds=old.folds
class QuantileSSM(old.QuantileSSM):
    """RevIN adapted to interpolation residuals, including positive learned affine."""
    def __init__(self,cfg,qd,hd):
        super().__init__(cfg,qd,hd)
        self.revin=cfg['mode']=='day_reversible'
        self.affine_log_scale=torch.nn.Parameter(torch.zeros(4),requires_grad=self.revin)
        self.affine_bias=torch.nn.Parameter(torch.zeros(4),requires_grad=self.revin)
    def affine_inputs(self,q,r):
        gamma=self.affine_log_scale.clamp(-3,3).exp()
        # The shared learned zone affine applies only to price-valued channels.
        # Absolute mean/std side channels, calendars and availability stay intact.
        qq=q.reshape(q.shape[0],24,4,27)
        fields=[]
        for k in range(27):
            x=qq[...,k]
            if k in [0,1,2,5,6,7,8] or k>=18:x=x*gamma[None,None,:]
            if k==0:x=x+self.affine_bias[None,None,:]
            fields.append(x)
        q=torch.stack(fields,-1).reshape(q.shape)
        chunks=[]
        for z in range(4):
            x=r[:,:,10*z:10*z+10]
            mask=x[...,8:9]
            chunks.append(torch.cat([x[...,:8]*gamma[z],mask,(x[...,9:10]*gamma[z]+self.affine_bias[z])*mask],-1))
        r=torch.cat(chunks+[r[...,40:]],-1)
        return q,r,gamma
    def forward(self,q,r):
        if not self.revin:return super().forward(q,r)
        q,r,gamma=self.affine_inputs(q,r)
        # The head is a deviation from interpolation: the anchor restores mean.
        # Divide every quantile by the same positive gamma to preserve ordering.
        return super().forward(q,r)/gamma[None,:,None,None,None]



def configs():
    return [dict(id=i,family=f,mode=m,width=32,state=32,lr=lr,seed=s)
            for i,(f,m,lr,s) in enumerate(itertools.product(FAMILIES,MODES,[.001,.003],[17,43]))]


def daily_inputs(q,r,h):
    """Price statistics are input-only and independent for each query day/zone."""
    mu=h.mean(-1);sd=h.std(-1);scale=np.sqrt(sd**2+1)
    qq=q.reshape(len(h),24,4,-1).transpose(0,2,1,3).copy()
    assert qq.shape[-1]==27
    qq[...,0]=(100*qq[...,0]-mu[:,:,None])/scale[:,:,None]
    qq[...,1:3]*=20/scale[:,:,None,None]
    # Channels 3/4 retain absolute mean/volatility side information.
    qq[...,5:9]*=20/scale[:,:,None,None]
    qq[...,18:]*=20/scale[:,:,None,None]
    rr=r.copy().reshape(len(h),168,43)
    for z in range(4):
        rr[:,:,10*z:10*z+8]*=20/scale[:,z,None,None]
        rr[:,:,10*z+9]=(100*rr[:,:,10*z+9]-mu[:,z,None])/scale[:,z,None]
        rr[:,:,10*z:10*z+10]*=rr[:,:,10*z+8,None]
    return qq.transpose(0,2,1,3).reshape(q.shape),rr,scale


def transform(q,r,h,nt,mode):
    scale=np.full(h.shape[:2],20.)
    if mode.startswith('day_'):
        q,r,s=daily_inputs(q,r,h)
        if mode=='day_reversible':scale=s
    if mode in ['fixed','day_reversible']:
        mq=np.zeros((1,1,q.shape[-1]));sq=np.ones_like(mq)
        mr=np.zeros((1,1,r.shape[-1]));sr=np.ones_like(mr)
        q=np.clip(q,-15,15).astype('float32');r=np.clip(r,-15,15).astype('float32')
    else:
        q,mq,sq=old.normalize(q,nt)
        r,mr,sr=old.normalize(r,nt,True)
    return dict(q=q,r=r,muq=mq,sdq=sq,mur=mr,sdr=sr,scale=scale)


def physical(raw,anchor,scale):
    raw=np.asarray(raw,dtype='float64')
    raw-=raw[...,5].mean(-1,keepdims=True)[...,None]
    result=anchor[...,None]+raw*scale[:,:,None,None,None]
    assert np.isfinite(result).all() and np.diff(result,axis=-1).min()>=-1e-8
    return result


def prepare():
    assert torch.cuda.is_available(),'Use the authorized remote GPU'
    ROOT.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256((PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==SHA
    dates,y=read_targets(PROJECT/'train_prices_15min.csv');h=y.mean(-1);b=baseline(h)
    q,r=old.features(dates,h,dates,y)
    for i in [0,91,120,302,333]:
        changed=y.copy();changed[i:]+=np.arange(4)*123456
        a,c=old.features(dates[i:i+1],h[i:i+1],dates,changed)
        u,v=old.features(dates[i:i+1],h[i:i+1],dates[:i],y[:i])
        np.testing.assert_allclose(a,q[i:i+1],atol=1e-9,rtol=0)
        np.testing.assert_allclose(c,r[i:i+1],atol=1e-9,rtol=0)
        np.testing.assert_allclose(a,u,atol=1e-9,rtol=0)
        np.testing.assert_allclose(c,v,atol=1e-9,rtol=0)
    paths=list(json.loads((PROJECT/'results/ssm_monthly_20261017/manifest.json').read_text())['code_hashes'])
    paths += [str((HERE/f).relative_to(PROJECT)) for f in ['train.py','audit.py','run.py','PROTOCOL.md','diagnostics.py']]
    manifest=dict(revin_affine=True,source_sha256=SHA,configs=configs(),folds=folds(dates),taus=TAUS.tolist(),
                  code_hashes={p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in paths},
                  causal_perturbation_and_truncation=True,physical_currency_loss=True)
    if (ROOT/'manifest.json').exists():assert manifest==json.loads((ROOT/'manifest.json').read_text())
    write(ROOT/'manifest.json',manifest)
    np.savez(ROOT/'raw.npz',dates=dates.astype('U10'),y=y,h=h,b=b,q=q,r=r)
    print('PREPARED',len(configs()),'fits per fold',flush=True)


def folder_data(fold,smoke=False):
    raw=np.load(ROOT/'raw.npz');n=sum(fold[k] for k in ['nt','ns','nc','test_days']);nt=fold['nt']
    folder=ROOT/('smoke' if smoke else 'folds')/fold['test'];folder.mkdir(parents=True,exist_ok=True)
    data={k:raw[k][:n] for k in ['dates','y','h','b']}
    data.update(nt=nt,ns=fold['ns'],nc=fold['nc'],selection_interpolation_mae=float(np.abs(raw['y'][nt:nt+fold['ns']]-raw['b'][nt:nt+fold['ns']]).mean()))
    for mode in MODES:
        x=transform(raw['q'][:n],raw['r'][:n],raw['h'][:n],nt,mode)
        data.update({mode+'_'+k:v for k,v in x.items()})
    np.savez(folder/'data.npz',**data);write(folder/'manifest.json',fold)
    return folder


def inputs(d,mode):
    return d[mode+'_q'],d[mode+'_r'],d[mode+'_scale']


def predict(net,d,mode,nt):
    q,r,s=inputs(d,mode);out=[]
    with torch.no_grad():
        for start in range(nt,len(d['y']),32):
            out.append(net(torch.tensor(q[start:start+32],device='cuda'),torch.tensor(r[start:start+32],device='cuda')).cpu().numpy())
    return physical(np.concatenate(out),d['b'][nt:],s[nt:])


def fit(cfg,folder,smoke=False):
    setup(cfg['seed']);folder=Path(folder);dest=folder/'trials'/str(cfg['id']);start=time.perf_counter()
    d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns']);x,z,scale=inputs(d,cfg['mode'])
    net=QuantileSSM(cfg,x.shape[-1],z.shape[-1])
    net.initialize((d['y'][:nt]-d['b'][:nt])/scale[:nt,:,None,None]);net.cuda()
    q=torch.tensor(x[:nt+ns],device='cuda');r=torch.tensor(z[:nt+ns],device='cuda')
    target=torch.tensor((d['y'][:nt+ns]-d['b'][:nt+ns])/20,dtype=torch.float32,device='cuda')
    multiplier=torch.tensor(scale[:nt+ns,:,None,None,None]/20,dtype=torch.float32,device='cuda')
    tau=torch.tensor(TAUS,dtype=torch.float32,device='cuda');opt=old.optimizer(net,cfg['lr'])
    best=float('inf');be=0;state=None;curves=[]
    for epoch in range(1,(2 if smoke else 300)+1):
        net.train()
        for ids in torch.randperm(nt,device='cuda').split(32):
            out=net(q[ids],r[ids])*multiplier[ids];err=target[ids,...,None]-out
            loss=err[...,5].abs().mean()+torch.maximum(tau*err,(tau-1)*err).mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():
            out=net(q[nt:],r[nt:])*multiplier[nt:];err=target[nt:,...,None]-out
            val=float(err[...,5].abs().mean())*20;pb=float(torch.maximum(tau*err,(tau-1)*err).mean())*20
        curves.append([epoch,val/float(d['selection_interpolation_mae']),pb]);assert np.isfinite(val+pb)
        if val<best:best=val;be=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-be>=35:break
    net.load_state_dict(state);net.eval();quantiles=predict(net,d,cfg['mode'],nt)
    dest.mkdir(parents=True,exist_ok=True);torch.save(dict(config=cfg,state=state),dest/'model.pt')
    np.save(dest/'quantiles.npy',quantiles);np.savetxt(dest/'curves.csv',curves,delimiter=',',header='epoch,selection_ratio,pinball',comments='')
    val=float(np.abs(quantiles[:ns,...,5]-d['y'][nt:nt+ns]).mean())
    row=dict(config=cfg,validation_mae=val,selection_ratio=val/float(d['selection_interpolation_mae']),best_curve_ratio=best/float(d['selection_interpolation_mae']),best_epoch=be,epochs=epoch,
             parameters=sum(p.numel() for p in net.parameters()),seconds=time.perf_counter()-start)
    write(dest/'metrics.json',row);return row


def select(folder,rows):
    winners={recipe:min([r for r in rows if r['config']['family']+'_'+r['config']['mode']==recipe],key=lambda r:(r['selection_ratio'],r['config']['id'])) for recipe in RECIPES}
    winner=min(winners,key=lambda k:(winners[k]['selection_ratio'],k))
    winners['validation_selector']=dict(winners[winner],chosen_recipe=winner)
    d=np.load(folder/'data.npz');nt=int(d['nt']);emp=np.quantile(d['y'][:nt]-d['b'][:nt],TAUS,axis=(0,2)).transpose(1,2,0)
    emp[...,:5]=np.minimum(emp[...,:5],0);emp[...,5]=0;emp[...,6:]=np.maximum(emp[...,6:],0)
    dest=folder/'trials/interpolation';dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'quantiles.npy',d['b'][nt:,...,None]+emp[None,:,None,:,:])
    winners['interpolation']=dict(config=dict(id='interpolation',family='interpolation'),selection_ratio=1.,validation_mae=float(d['selection_interpolation_mae']))
    write(folder/'selection.json',dict(by_recipe=winners,candidates=rows));return winners


def run(smoke=False):
    raw=np.load(ROOT/'raw.npz');jobs=folds(raw['dates']);reports=[];paired=[]
    if smoke:jobs=[jobs[0],jobs[-1]]
    for fold in jobs:
        folder=folder_data(fold,smoke);candidates=configs()
        if smoke:candidates=[c for c in candidates if c['lr']==.001 and c['seed']==17]
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for fut in cf.as_completed([pool.submit(fit,c,str(folder),smoke) for c in candidates]):
                row=fut.result();rows.append(row);write(folder/'progress.json',dict(completed=len(rows),total=len(candidates)))
                print('FIT',fold['test'],len(rows),row,flush=True)
        winners=select(folder,rows);report=evaluation.evaluate(folder,winners)
        d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc;truth=d['y'][start:];hh=d['h'][start:]
        uncorrected=np.empty_like(truth)
        for i in np.ndindex(hh.shape[:-1]):uncorrected[i]=np.interp(np.arange(96)/4+.125,np.arange(24)+.5,hh[i]).reshape(24,4)
        alt=float(np.abs(uncorrected-truth).mean());zonebase=np.abs(d['b'][start:]-truth).mean((0,2,3));den=float(zonebase.mean())
        for score in report['results'].values():
            score.update(uncorrected_interpolation_mae=alt,ratio_to_uncorrected_interpolation=score['mae']/alt,
                         zone_interpolation_mae=zonebase.tolist(),zone_ratio=(np.asarray(score['by_zone'])/zonebase).tolist())
        for row in rows:
            cfg=row['config'];q=np.load(folder/'trials'/str(cfg['id'])/'quantiles.npy')[ns+nc:]
            mae=float(np.abs(q[...,5]-truth).mean())
            paired.append(dict(month=fold['test'],family=cfg['family'],mode=cfg['mode'],lr=cfg['lr'],seed=cfg['seed'],mae=mae,ratio=mae/den))
        write(folder/'metrics.json',report);reports.append(report);evaluation.summarize(reports,smoke)
        out=ROOT/'smoke' if smoke else ROOT
        pd.DataFrame(paired).to_csv(out/'all_candidate_scores.csv',index=False)
        summary=json.loads((out/'metrics.json').read_text());basepool=np.average([v['results']['interpolation']['mae'] for v in reports],weights=[v['manifest']['test_days'] for v in reports])
        for score in summary['summary']:score.update(pooled_interpolation_mae=float(basepool),pooled_mae_ratio=score['pooled_mae']/basepool)
        write(out/'metrics.json',summary)
        write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_folds=len(reports),total_folds=len(jobs),last_test_month=fold['test']))
    if not smoke:pd.concat([pd.read_csv(ROOT/'folds'/f['test']/'predictions.csv.gz') for f in jobs],ignore_index=True).to_csv(ROOT/'predictions.csv.gz',index=False,compression='gzip')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','run']);a=p.parse_args()
    prepare() if a.stage=='prepare' else run(a.stage=='smoke')
