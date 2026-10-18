"""All learned recipes jointly predict a conserved median and ordered quantiles."""
import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
BASE=PROJECT/'campaigns/monthly_ratio_20261017'
sys.path.insert(0,str(BASE))
import experiment as p
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=PROJECT/'results/monthly_quantiles_20261017'
POINT_ROOT=PROJECT/'results/monthly_ratio_20261017'
p.ROOT=ROOT
TAUS=np.array([.025,.05,.10,.20,.25,.50,.75,.80,.90,.95,.975])
INTERVALS=[(.50,4,6),(.80,2,8),(.90,1,9),(.95,0,10)]


def configs():
    return [dict(**c,quantile_weight=1.0) for c in p.configs()]


def prepare():
    p.setup();p.weighting_test();ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((POINT_ROOT/'manifest.json').read_text())
    assert old['source_sha256']==p.SHA and old['raw_feature_causality_passed']
    for path,sha in old['code_hashes'].items():
        assert hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==sha,path
    assert hashlib.sha256((PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==p.SHA
    paths=p.DEPENDENCIES+[str((HERE/name).relative_to(PROJECT)) for name in ['trainq.py','auditq.py','run.py','PROTOCOL.md']]
    manifest=dict(source_sha256=p.SHA,configs=configs(),taus=TAUS.tolist(),folds=old['folds'],
                  code_hashes={path:hashlib.sha256((PROJECT/path).read_bytes()).hexdigest() for path in paths},
                  feature_source='results/monthly_ratio_20261017/raw.npz',
                  feature_source_sha256=hashlib.sha256((POINT_ROOT/'raw.npz').read_bytes()).hexdigest(),
                  selection='preceding month days1-15 q50 MAE ratio; calibration16-end; outer test untouched')
    if (ROOT/'manifest.json').exists():
        assert json.loads((ROOT/'manifest.json').read_text())==manifest,'Source changed; use a new output directory'
    shutil.copy2(POINT_ROOT/'raw.npz',ROOT/'raw.npz')
    p.write(ROOT/'feature_source_manifest.json',old);p.write(ROOT/'manifest.json',manifest)
    print('PREPARED quantile campaign',len(configs()),'candidates/fold',flush=True)


class Quantiles(nn.Module):
    def __init__(self,cfg,data):
        super().__init__();self.family=cfg['family'];self.two_d=self.family=='cnn2d_analog'
        if self.two_d:
            body=p.Global2D();self.ch,self.aux=body.channels,body.aux
            self.conv,self.fusion,self.full_curve=body.conv,body.fusion,body.full_curve
            self.head=nn.Conv1d(16,44,1);self.feature='cnn'
        else:
            family,self.feature=self.family.split('_',1);n=data[self.feature].shape[-1]
            self.register_buffer('knots',torch.tensor(data[self.feature+'_knots']))
            if family in ['linear','additive']:
                self.head=nn.Linear(n*(4 if family=='additive' else 1),43)
            elif family=='mlp':
                self.body=nn.Sequential(nn.Linear(n,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
                self.head=nn.Linear(32,43)
            else:
                self.body=nn.Sequential(nn.Conv1d(n,16,5,padding=2,padding_mode='replicate'),nn.SiLU(),
                                        nn.Conv1d(16,16,5,padding=2,padding_mode='replicate'),nn.SiLU())
                self.head=nn.Conv1d(16,43,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def initialize(self,target):
        axes=(0,2,3) if self.two_d else (0,1,2)
        empirical=np.quantile(target,TAUS,axis=axes).T
        below=np.concatenate([np.maximum(-empirical[:,4:5],.005),np.maximum(np.diff(empirical[:,:5],axis=1)[:,::-1],.005)],-1)
        above=np.concatenate([np.maximum(empirical[:,6:7],.005),np.maximum(np.diff(empirical[:,6:],axis=1),.005)],-1)
        gaps=np.concatenate([below,above],-1)
        with torch.no_grad():
            self.head.bias[4 if self.two_d else 3:].copy_(torch.as_tensor(np.log(np.expm1(gaps)).ravel(),dtype=self.head.bias.dtype))

    def forward(self,x,curve):
        if self.two_d:
            grid=x[:,:-self.aux].reshape(-1,self.ch,16,96)
            encoded=F.relu(self.conv(F.pad(grid,(3,4,0,0),mode='replicate'))).flatten(1,2)
            fused=F.relu(self.fusion(torch.cat([encoded,x[:,-self.aux:]],1))+self.full_curve(curve)[:,:,None])
            raw=self.head(fused)
            median=raw[:,:4].reshape(-1,4,24,4)
            increments=raw[:,4:].reshape(-1,4,10,96).permute(0,1,3,2).reshape(-1,4,24,4,10)
        else:
            if self.family.startswith('additive'):
                x=torch.cat([x]+[(x-k).relu() for k in self.knots],-1)
            if self.family.startswith('cnn'):
                b,z,t,n=x.shape
                raw=self.head(self.body(x.reshape(b*z,t,n).transpose(1,2))).transpose(1,2).reshape(b,z,t,43)
            else:
                raw=self.head(self.body(x) if hasattr(self,'body') else x)
            a,u,v=raw[...,:3].unbind(-1)
            median=torch.stack([a+u,a-u,-a+v,-a-v],-1)
            increments=raw[...,3:].reshape(*raw.shape[:-1],4,10)
        median=median-median.mean(-1,keepdim=True)
        inc=F.softplus(increments)+1e-6
        lower=(median[...,None]-inc[...,:5].cumsum(-1)).flip(-1)
        upper=median[...,None]+inc[...,5:].cumsum(-1)
        return torch.cat([lower,median[...,None],upper],-1)


def physical(raw,anchor):
    raw=np.asarray(raw,dtype='float64')
    raw=raw-raw[...,5].mean(-1,keepdims=True)[...,None]
    result=anchor[...,None]+20*raw
    assert np.isfinite(result).all() and np.diff(result,axis=-1).min()>=-1e-9
    return result


def fit(cfg,folder,smoke=False):
    p.setup(cfg['seed']);folder=Path(folder);dest=folder/'trials'/str(cfg['id']);start=time.time()
    if (dest/'metrics.json').exists() and (dest/'quantiles.npy').exists():
        row=json.loads((dest/'metrics.json').read_text());assert row['config']==cfg;return row
    d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns'])
    net=Quantiles(cfg,d);net.initialize((d['y'][:nt]-d['b'][:nt])/20);net.cuda()
    feature=net.feature;x=torch.as_tensor(d[feature][:nt+ns],device='cuda');curve=torch.as_tensor(d['curve'][:nt+ns],device='cuda')
    target=torch.tensor((d['y'][:nt+ns]-d['b'][:nt+ns])/20,dtype=torch.float32,device='cuda')
    weights=torch.as_tensor(d['weights_'+cfg['objective']],device='cuda')
    tau=torch.tensor(TAUS,dtype=torch.float32,device='cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01)
    best,best_epoch,state=float('inf'),0,None;curves=[]
    limit=2 if smoke else 450 if net.two_d else 300;patience=50 if net.two_d else 35
    for epoch in range(1,limit+1):
        net.train()
        for ids in torch.randperm(nt,device='cuda').split(32):
            q=net(x[ids],curve[ids]);err=target[ids,...,None]-q
            mae=err[...,5].abs().mean((1,2,3));pb=torch.maximum(tau*err,(tau-1)*err).mean((1,2,3,4))
            loss=((mae+cfg['quantile_weight']*pb)*weights[ids]).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
        net.eval()
        with torch.no_grad():
            q=net(x[nt:],curve[nt:]);value=float((q[...,5]-target[nt:]).abs().mean())*20
            err=target[nt:,...,None]-q;pb=float(torch.maximum(tau*err,(tau-1)*err).mean())*20
        assert np.isfinite(value) and np.isfinite(pb)
        curves.append([epoch,value/float(d['selection_interpolation_mae']),pb])
        if value<best:
            best,best_epoch=value,epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch-best_epoch>=patience:break
    net.load_state_dict(state);net.eval()
    ex=torch.as_tensor(d[feature][nt:],device='cuda');ec=torch.as_tensor(d['curve'][nt:],device='cuda')
    with torch.no_grad():raw=net(ex,ec).cpu().numpy()
    q=physical(raw,d['b'][nt:]);assert np.abs(q[...,5].mean(-1)-d['h'][nt:]).max()<1e-8
    dest.mkdir(parents=True,exist_ok=True);torch.save(dict(config=cfg,state=state,feature=feature),dest/'model.pt')
    np.save(dest/'quantiles.npy',q);np.savetxt(dest/'curves.csv',curves,delimiter=',',header='epoch,selection_ratio,pinball_mae_units',comments='')
    replay=Quantiles(cfg,d).cuda().eval();replay.load_state_dict(torch.load(dest/'model.pt',weights_only=False,map_location='cpu')['state'])
    with torch.no_grad():rr=physical(replay(ex,ec).cpu().numpy(),d['b'][nt:])
    delta=float(np.abs(rr-q).max());assert delta<1e-6
    val=float(np.abs(q[:ns,...,5]-d['y'][nt:nt+ns]).mean())
    row=dict(config=cfg,validation_mae=val,selection_ratio=val/float(d['selection_interpolation_mae']),
             parameters=sum(v.numel() for v in net.parameters()),best_epoch=best_epoch,epochs=epoch,seconds=time.time()-start,replay_max_error=delta)
    p.write(dest/'metrics.json',row);return row


def train(folder,smoke=False):
    jobs=configs()
    if smoke:jobs=[next(c for c in jobs if c['family']==f and c['objective']==o) for f,o in itertools.product(p.FAMILIES,['uniform','ratio'])]
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        for future in cf.as_completed([pool.submit(fit,c,str(folder),smoke) for c in jobs]):
            row=future.result();rows.append(row);p.write(folder/'progress.json',dict(completed=len(rows),total=len(jobs),latest=row))
            print('QUANTILE FIT',folder.name,len(rows),len(jobs),row,flush=True)
    return rows


def select(folder,rows):
    d=np.load(folder/'data.npz');nt,ns=int(d['nt']),int(d['ns']);winners={}
    for family,objective in itertools.product(p.FAMILIES,['uniform','ratio']):
        subset=[r for r in rows if r['config']['family']==family and r['config']['objective']==objective]
        winners[family+'_'+objective]=min(subset,key=lambda r:(r['selection_ratio'],r['config']['id']))
    for objective in ['uniform','ratio']:
        members={name:row for name,row in winners.items() if row['config']['objective']==objective}
        predictions={name:np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy') for name,row in members.items()}
        primary='linear_rich_'+objective
        choices=[{name:1.} for name in sorted(members)]
        choices += [{primary:w,other:1-w} for other in sorted(members) if other!=primary for w in [.25,.5,.75]]
        top=sorted(members,key=lambda name:(members[name]['selection_ratio'],name))[:3];choices.append({name:1/3 for name in top})
        candidates=[]
        for weights in choices:
            q=sum(predictions[name][:ns]*w for name,w in weights.items())
            ratio=float(np.abs(q[...,5]-d['y'][nt:nt+ns]).mean())/float(d['selection_interpolation_mae'])
            candidates.append(dict(weights=weights,selection_ratio=ratio))
        chosen=min(candidates,key=lambda v:v['selection_ratio']);cid='blend_'+objective
        path=folder/'trials'/cid;path.mkdir(parents=True,exist_ok=True)
        np.save(path/'quantiles.npy',sum(predictions[name]*w for name,w in chosen['weights'].items()))
        winners[cid]=dict(config=dict(id=cid,family='blend',objective=objective),selection_ratio=chosen['selection_ratio'],
                          validation_mae=chosen['selection_ratio']*float(d['selection_interpolation_mae']),
                          weights=chosen['weights'],blend_candidates=candidates)
    residual=d['y'][:nt]-d['b'][:nt]
    empirical=np.quantile(residual,TAUS,axis=(0,2)).transpose(1,2,0)  # zone,quarter,quantile
    empirical[...,:5]=np.minimum(empirical[...,:5],0);empirical[...,5]=0;empirical[...,6:]=np.maximum(empirical[...,6:],0)
    q=d['b'][nt:,...,None]+empirical[None,:,None,:,:]
    dest=folder/'trials'/'interpolation';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',q)
    winners['interpolation']=dict(config=dict(id='interpolation',family='interpolation',objective='empirical_training_quantiles'),
                                  selection_ratio=1.0,validation_mae=float(d['selection_interpolation_mae']))
    chosen=min(winners,key=lambda name:(winners[name]['selection_ratio'],name))
    winners['validation_selector']=dict(winners[chosen],chosen_recipe=chosen)
    p.write(folder/'selection.json',dict(by_recipe=winners,candidates=rows))
    return winners


def intervals(cal_q,cal_y,test_q):
    """Widen-only finite-sample CQR, then deterministic nesting."""
    result={};previous_lower=previous_upper=None;previous_cal_lower=previous_cal_upper=None
    for coverage,lo,hi in INTERVALS:
        scores=np.maximum.reduce([cal_q[...,lo]-cal_y,cal_y-cal_q[...,hi],np.zeros_like(cal_y)])
        pooled=scores.transpose(0,2,1,3).reshape(-1,4,4);rank=math.ceil((len(pooled)+1)*coverage)
        assert rank<=len(pooled)
        correction=np.sort(pooled,axis=0)[rank-1][None,:,None,:]
        lower=test_q[...,lo]-correction;upper=test_q[...,hi]+correction
        cal_lower=cal_q[...,lo]-correction;cal_upper=cal_q[...,hi]+correction
        if previous_lower is not None:
            lower=np.minimum(lower,previous_lower);upper=np.maximum(upper,previous_upper)
            cal_lower=np.minimum(cal_lower,previous_cal_lower);cal_upper=np.maximum(cal_upper,previous_cal_upper)
        level=round(coverage*100)
        result[level]=dict(lower=lower,upper=upper,cal_lower=cal_lower,cal_upper=cal_upper,
                           correction=correction[0,:,0,:],rank=rank,n=len(pooled),coverage=coverage)
        previous_lower,previous_upper,previous_cal_lower,previous_cal_upper=lower,upper,cal_lower,cal_upper
    return result


def percentile(width,cal_width):
    out=np.empty_like(width)
    for z,quarter in itertools.product(range(4),range(4)):
        reference=np.sort(cal_width[:,z,:,quarter].ravel())
        values=width[:,z,:,quarter]
        out[:,z,:,quarter]=(np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right'))/(2*len(reference))
    return out


def evaluate(folder,winners):
    d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc'])
    truth=d['y'][nt+ns+nc:];base=d['b'][nt+ns+nc:];baseline=float(np.abs(base-truth).mean());results={}
    for recipe,row in winners.items():
        q=np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy')
        cal_q=q[ns:ns+nc];test_q=q[ns+nc:]
        bands=intervals(cal_q,d['y'][nt+ns:nt+ns+nc],test_q)
        path=folder/'selected'/recipe;path.mkdir(parents=True,exist_ok=True)
        p.write(path/'calibration.json',{str(k):{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in item.items() if key not in ['lower','upper','cal_lower','cal_upper']} for k,item in bands.items()})
        # Calibration quantities above are frozen before test-error diagnostics.
        error=np.abs(test_q[...,5]-truth);pb_error=truth[...,None]-test_q
        pinball=np.maximum(TAUS*pb_error,(TAUS-1)*pb_error).mean((0,1,2,3))
        width=bands[90]['upper']-bands[90]['lower'];cal_width=bands[90]['cal_upper']-bands[90]['cal_lower']
        pct=percentile(width,cal_width);labels=np.where(pct<=.25,0,np.where(pct>=.75,2,1)).astype('int8')
        saved=dict(dates=d['dates'][nt+ns+nc:],observed=truth,quantiles=test_q,taus=TAUS,
                   uncertainty_percentile=pct,confidence_bin=labels)
        summaries=[]
        for coverage,lo,hi in INTERVALS:
            level=round(coverage*100);band=bands[level]
            saved[f'lower_{level}']=band['lower'];saved[f'upper_{level}']=band['upper']
            for kind,lower,upper in [('raw',test_q[...,lo],test_q[...,hi]),('calibrated',band['lower'],band['upper'])]:
                covered=(truth>=lower)&(truth<=upper)
                summaries.append(dict(kind=kind,nominal=coverage,coverage=float(covered.mean()),width=float((upper-lower).mean()),
                                      by_zone=covered.mean((0,2,3)).tolist(),whole_hour_coverage=float(covered.all(-1).mean()),
                                      whole_day_coverage=float(covered.all((1,2,3)).mean())))
        bins=[]
        for index,label in enumerate(['narrow','middle','wide']):
            mask=labels==index;count=int(mask.sum())
            bins.append(dict(label=label,n=count,mae=float(error[mask].mean()) if count else None,
                             interpolation_mae=float(np.abs(base-truth)[mask].mean()) if count else None,
                             ratio=float(error[mask].mean()/np.abs(base-truth)[mask].mean()) if count and np.abs(base-truth)[mask].mean()>0 else None,
                             coverage90=float(((truth>=bands[90]['lower'])&(truth<=bands[90]['upper']))[mask].mean()) if count else None,
                             width90=float(width[mask].mean()) if count else None))
        correlation=pd.Series(width.ravel()).rank().corr(pd.Series(error.ravel()).rank())
        results[recipe]=dict(mae=float(error.mean()),interpolation_mae=baseline,ratio=float(error.mean())/baseline,
                             by_zone=error.mean((0,2,3)).tolist(),pinball_by_quantile=pinball.tolist(),
                             mean_pinball=float(pinball.mean()),pinball_ratio=float(pinball.mean())/baseline,
                             intervals=summaries,confidence_bins=bins,width_error_spearman=float(correlation) if np.isfinite(correlation) else None,
                             max_hourly_mean_error=float(np.abs(test_q[...,5].mean(-1)-d['h'][nt+ns+nc:]).max()))
        assert results[recipe]['max_hourly_mean_error']<1e-8
        np.savez_compressed(path/'predictions.npz',**saved)
        if recipe=='validation_selector':
            n=len(truth);dd,zz,hh,qq=np.indices((n,4,24,4))
            columns=dict(delivery_date=saved['dates'][dd.ravel()],market_id=np.array([f'ZONE_{z}' for z in range(4)])[zz.ravel()],
                         hour_index=hh.ravel()+1,quarter_index=qq.ravel()+1)
            for k,tau in enumerate(TAUS):columns[f'q{tau:g}']=test_q[...,k].ravel()
            columns.update(lower_90=bands[90]['lower'].ravel(),upper_90=bands[90]['upper'].ravel(),
                           width_90=width.ravel(),uncertainty_percentile=pct.ravel(),confidence_bin=np.array(['narrow','middle','wide'])[labels.ravel()])
            pd.DataFrame(columns).to_csv(folder/'predictions.csv.gz',index=False,float_format='%.8f',compression='gzip')
    report=dict(manifest=json.loads((folder/'manifest.json').read_text()),selected=winners,results=results)
    p.write(folder/'metrics.json',report);return report


def summarize(reports,smoke=False):
    out=ROOT/'smoke' if smoke else ROOT;rows=[];interval_rows=[];bin_rows=[]
    for report in reports:
        meta=report['manifest']
        for recipe,score in report['results'].items():
            rows.append(dict(month=meta['test'],recipe=recipe,test_days=meta['test_days'],mae=score['mae'],ratio=score['ratio'],
                             mean_pinball=score['mean_pinball'],pinball_ratio=score['pinball_ratio']))
            interval_rows += [dict(month=meta['test'],recipe=recipe,**v) for v in score['intervals']]
            bin_rows += [dict(month=meta['test'],recipe=recipe,**v) for v in score['confidence_bins']]
    frame=pd.DataFrame(rows);frame.to_csv(out/'monthly_scores.csv',index=False)
    pd.DataFrame(interval_rows).to_csv(out/'interval_metrics.csv',index=False);pd.DataFrame(bin_rows).to_csv(out/'confidence_bins.csv',index=False)
    summaries=[]
    for recipe,group in frame.groupby('recipe'):
        matching=[v for v in interval_rows if v['recipe']==recipe and v['kind']=='calibrated' and v['nominal']==.9]
        days=[next(r['manifest']['test_days'] for r in reports if r['manifest']['test']==v['month']) for v in matching]
        summaries.append(dict(recipe=recipe,folds=len(group),mean_monthly_ratio=float(group.ratio.mean()),
                              pooled_mae=float(np.average(group.mae,weights=group.test_days)),mean_monthly_pinball_ratio=float(group.pinball_ratio.mean()),
                              coverage90=float(np.average([v['coverage'] for v in matching],weights=days)),
                              width90=float(np.average([v['width'] for v in matching],weights=days)),worst_ratio=float(group.ratio.max())))
    summaries.sort(key=lambda v:v['mean_monthly_ratio']);pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    p.write(out/'metrics.json',dict(smoke=smoke,completed_folds=len(reports),summary=summaries))


def run(smoke=False):
    p.setup();d=np.load(ROOT/'raw.npz');jobs=p.folds(d['dates'])
    if smoke:jobs=[jobs[0],jobs[-1]]
    reports=[]
    for fold in jobs:
        folder=p.fold_data(fold,smoke);rows=train(folder,smoke);winners=select(folder,rows)
        reports.append(evaluate(folder,winners));summarize(reports,smoke)
        p.write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_folds=len(reports),total_folds=len(jobs),last_test_month=fold['test']))
        print('QUANTILE FOLD DONE',len(reports),len(jobs),fold['test'],flush=True)
    if smoke:p.write(ROOT/'smoke.json',dict(passed=True,folds=2,neural_fits=24,all_models_have_trained_quantiles=True,levels=TAUS.tolist()))
    else:
        pd.concat([pd.read_csv(ROOT/'folds'/fold['test']/'predictions.csv.gz') for fold in jobs],ignore_index=True).to_csv(ROOT/'predictions.csv.gz',index=False,compression='gzip',float_format='%.8f')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','smoke','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else run(args.stage=='smoke')
