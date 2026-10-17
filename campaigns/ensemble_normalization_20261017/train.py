"""Fixed-vs-standard scaling for the actual six experts behind the .7205 blend."""
import argparse,concurrent.futures as cf,hashlib,itertools,json,multiprocessing as mp,os,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'campaigns/monthly_quantiles_20261017'))
import trainq as e
import numpy as np
import pandas as pd
import torch
ROOT=PROJECT/'results/ensemble_normalization_20261017'
SOURCE=PROJECT/'results/monthly_quantiles_20261017'
MODES=['standard','fixed'];FAMILIES=e.p.FAMILIES;TAUS=e.TAUS
e.ROOT=ROOT;e.p.ROOT=ROOT


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def link(source,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():os.link(source,dest)


def configs():
    controls=[c for c in e.configs() if c['objective']=='uniform']
    return [dict(**{**c,'id':c['id']+(1000 if m=='fixed' else 0)},mode=m,reference_id=c['id']) for m,c in itertools.product(MODES,controls)]


def prepare():
    e.p.setup();ROOT.mkdir(parents=True,exist_ok=True)
    original=json.loads((SOURCE/'manifest.json').read_text())
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    for path,s in original['code_hashes'].items():assert sha(PROJECT/path)==s,path
    assert sha(PROJECT/'train_prices_15min.csv')==e.p.SHA
    assert sha(SOURCE/'raw.npz')==original['feature_source_sha256']
    paths=list(original['code_hashes'])+[str((HERE/name).relative_to(PROJECT)) for name in ['train.py','audit.py','run.py','PROTOCOL.md']]
    manifest=dict(source_sha256=e.p.SHA,code_hashes={p:sha(PROJECT/p) for p in paths},configs=configs(),folds=original['folds'],
                  feature_source_sha256=original['feature_source_sha256'],original_manifest_sha256=sha(SOURCE/'manifest.json'),
                  source_audit_sha256=sha(SOURCE/'audit.json'),new_fits=192,reused_controls=192,original_blend_ratio=.7204941118854624)
    if (ROOT/'manifest.json').exists():assert json.loads((ROOT/'manifest.json').read_text())==manifest
    e.p.write(ROOT/'manifest.json',manifest);link(SOURCE/'raw.npz',ROOT/'raw.npz')
    print('PREPARED stronger ensemble normalization,192 new fits and192 reused controls',flush=True)


def folder_data(fold,smoke=False):
    folder=ROOT/('smoke' if smoke else 'folds')/fold['test'];folder.mkdir(parents=True,exist_ok=True)
    source=SOURCE/'folds'/fold['test'];link(source/'data.npz',folder/'data.npz')
    meta=json.loads((source/'manifest.json').read_text());e.p.write(folder/'manifest.json',meta)
    d=np.load(folder/'data.npz');raw=np.load(ROOT/'raw.npz');n=len(d['dates']);nt=int(d['nt'])
    arrays={k:d[k] for k in d.files if k not in {f+s for f in ['rich','compact','cnn','curve'] for s in ['','_mu','_sd','_knots']}}
    for feature in ['rich','compact','cnn','curve']:
        x=np.clip(raw[feature][:n],-15,15).astype('float32')
        if feature=='compact':x=np.concatenate([x,np.broadcast_to(np.eye(4)[None,:,None,:],(n,4,24,4))],-1).astype('float32')
        axes=tuple(range(raw[feature].ndim-1)) if feature in ['rich','compact'] else (0,2) if feature=='cnn' else (0,)
        shape=list(raw[feature].shape)
        for axis in axes:shape[axis]=1
        arrays.update({feature:x,feature+'_mu':np.zeros(shape),feature+'_sd':np.ones(shape)})
        if feature in ['rich','compact']:arrays[feature+'_knots']=np.quantile(x[:nt].reshape(-1,x.shape[-1]),[.25,.5,.75],axis=0).astype('float32')
    fixed=folder/'fixed';fixed.mkdir(exist_ok=True);np.savez(fixed/'data.npz',**arrays)
    e.p.write(fixed/'manifest.json',dict(**meta,preprocessing='fixed engineered units, no fitted z-score'))
    return folder


def fit(cfg,folder,smoke=False):
    folder=Path(folder);mode=cfg['mode'];dest=folder/'trials'/str(cfg['id'])
    if mode=='standard' and not smoke:
        source=SOURCE/'folds'/folder.name/'trials'/str(cfg['reference_id'])
        row=json.loads((source/'metrics.json').read_text());original={k:v for k,v in cfg.items() if k not in ['mode','reference_id']}
        assert row['config']==original
        ck=torch.load(source/'model.pt',weights_only=False,map_location='cpu');assert ck['config']==original
        dest.mkdir(parents=True,exist_ok=True);torch.save(dict(**{**ck,'config':cfg}),dest/'model.pt')
        for name in ['quantiles.npy','curves.csv']:link(source/name,dest/name)
        row['config']=cfg;row['reused_control']=True
        e.p.write(dest/'reference.json',dict(source=str(source.relative_to(PROJECT)),model_sha256=sha(source/'model.pt'),prediction_sha256=sha(source/'quantiles.npy')))
        e.p.write(dest/'metrics.json',row);return row
    target=folder/'fixed' if mode=='fixed' else folder
    row=e.fit(cfg,target,smoke)
    if target!=folder:
        dest.parent.mkdir(exist_ok=True);shutil.move(str(target/'trials'/str(cfg['id'])),str(dest))
    row['reused_control']=False;e.p.write(dest/'metrics.json',row);return row


def make_blend(folder,d,winners,members,name):
    ns,nt=int(d['ns']),int(d['nt']);pred={k:np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy') for k,row in members.items()}
    primary=next(k for k,v in members.items() if v['config']['family']=='linear_rich')
    choices=[{k:1.} for k in sorted(members)]
    choices += [{primary:w,other:1-w} for other in sorted(members) if other!=primary for w in [.25,.5,.75]]
    top=sorted(members,key=lambda k:(members[k]['selection_ratio'],k))[:3];choices.append({k:1/3 for k in top})
    candidates=[]
    for weights in choices:
        q=sum(pred[k][:ns]*w for k,w in weights.items());ratio=float(np.abs(q[...,5]-d['y'][nt:nt+ns]).mean())/float(d['selection_interpolation_mae'])
        candidates.append(dict(weights=weights,selection_ratio=ratio))
    chosen=min(candidates,key=lambda v:v['selection_ratio']);dest=folder/'trials'/name;dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'quantiles.npy',sum(pred[k]*w for k,w in chosen['weights'].items()))
    winners[name]=dict(config=dict(id=name,family='blend',mode=name.split('__')[-1],objective='uniform'),
                       selection_ratio=chosen['selection_ratio'],validation_mae=chosen['selection_ratio']*float(d['selection_interpolation_mae']),
                       weights=chosen['weights'],blend_candidates=candidates)


def select(folder,rows):
    d=np.load(folder/'data.npz');nt=int(d['nt']);winners={}
    for family,mode in itertools.product(FAMILIES,MODES):
        winners[family+'__'+mode]=min([r for r in rows if r['config']['family']==family and r['config']['mode']==mode],key=lambda r:(r['selection_ratio'],r['config']['id']))
    for mode in MODES:make_blend(folder,d,winners,{k:v for k,v in winners.items() if v['config']['mode']==mode},'blend__'+mode)
    mixed={}
    for family in FAMILIES:
        name=min([family+'__'+mode for mode in MODES],key=lambda k:(winners[k]['selection_ratio'],k));mixed[name]=winners[name]
    make_blend(folder,d,winners,mixed,'blend__mixed')
    empirical=np.quantile(d['y'][:nt]-d['b'][:nt],TAUS,axis=(0,2)).transpose(1,2,0)
    empirical[...,:5]=np.minimum(empirical[...,:5],0);empirical[...,5]=0;empirical[...,6:]=np.maximum(empirical[...,6:],0)
    dest=folder/'trials/interpolation';dest.mkdir(parents=True,exist_ok=True);np.save(dest/'quantiles.npy',d['b'][nt:,...,None]+empirical[None,:,None,:,:])
    winners['interpolation']=dict(config=dict(id='interpolation',family='interpolation'),selection_ratio=1.,validation_mae=float(d['selection_interpolation_mae']))
    chosen=min(winners,key=lambda k:(winners[k]['selection_ratio'],k));winners['validation_selector']=dict(winners[chosen],chosen_recipe=chosen)
    e.p.write(folder/'selection.json',dict(by_recipe=winners,candidates=rows));return winners


def run(smoke=False):
    jobs=json.loads((ROOT/'manifest.json').read_text())['folds'];reports=[];paired=[]
    if smoke:jobs=[jobs[0],jobs[-1]]
    for fold in jobs:
        folder=folder_data(fold,smoke);candidates=configs()
        if smoke:candidates=[c for c in candidates if c['lr']==.001 and c['seed']==17]
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for future in cf.as_completed([pool.submit(fit,c,str(folder),smoke) for c in candidates]):
                row=future.result();rows.append(row);e.p.write(folder/'progress.json',dict(completed=len(rows),total=len(candidates)))
                print('FIT',fold['test'],len(rows),row,flush=True)
        winners=select(folder,rows);report=e.evaluate(folder,winners)
        d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc;truth=d['y'][start:];hh=d['h'][start:]
        uncorrected=np.asarray([[np.interp(np.arange(96)/4+.125,np.arange(24)+.5,z).reshape(24,4) for z in day] for day in hh])
        alt=float(np.abs(uncorrected-truth).mean());zb=np.abs(d['b'][start:]-truth).mean((0,2,3));den=float(zb.mean())
        for score in report['results'].values():score.update(uncorrected_interpolation_mae=alt,ratio_to_uncorrected_interpolation=score['mae']/alt,zone_interpolation_mae=zb.tolist(),zone_ratio=(np.asarray(score['by_zone'])/zb).tolist())
        for row in rows:
            cfg=row['config'];q=np.load(folder/'trials'/str(cfg['id'])/'quantiles.npy')[ns+nc:];mae=float(np.abs(q[...,5]-truth).mean())
            paired.append(dict(month=fold['test'],family=cfg['family'],mode=cfg['mode'],lr=cfg['lr'],seed=cfg['seed'],mae=mae,ratio=mae/den))
        e.p.write(folder/'metrics.json',report);reports.append(report);e.summarize(reports,smoke)
        out=ROOT/'smoke' if smoke else ROOT;pd.DataFrame(paired).to_csv(out/'all_candidate_scores.csv',index=False)
        summary=json.loads((out/'metrics.json').read_text());bp=np.average([r['results']['interpolation']['mae'] for r in reports],weights=[r['manifest']['test_days'] for r in reports])
        for score in summary['summary']:score.update(pooled_interpolation_mae=float(bp),pooled_mae_ratio=score['pooled_mae']/bp)
        e.p.write(out/'metrics.json',summary);e.p.write(ROOT/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_folds=len(reports),total_folds=len(jobs),last_test_month=fold['test']))
    if not smoke:pd.concat([pd.read_csv(ROOT/'folds'/f['test']/'predictions.csv.gz') for f in jobs],ignore_index=True).to_csv(ROOT/'predictions.csv.gz',index=False,compression='gzip')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','run']);a=p.parse_args();prepare() if a.stage=='prepare' else run(a.stage=='smoke')
