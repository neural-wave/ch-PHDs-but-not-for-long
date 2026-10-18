"""Six-family expanding evaluation with isolated selection and conformal calibration."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from simple_conformal import calibrate,self_test

OUT=Path('results/expanding_simple_5090')

def folds(dates):
    months=sorted(set(str(d)[:7] for d in dates));assert len(months)==12
    result=[dict(plan='expanding',months=i+1,train_start=months[0],train_end=months[i],validation=months[i+1],test=months[i+2]) for i in range(2,10)]
    result += [dict(plan='fixed_september',months=s,train_start=months[10-s],train_end=months[9],validation=months[10],test=months[11]) for s in range(3,11)]
    return result

def load_raw():
    from reconstruction import read_targets,rich_features,baseline
    from interpretable_models import compact_features
    source=Path('train_prices_15min.csv');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    dates,y=read_targets(source);h=y.mean(-1);b=baseline(h)
    raw=dict(rich=rich_features(dates,h,dates,y).reshape(len(dates),4,24,-1),compact=compact_features(dates,h,dates,y,True)[0])
    return dates,y,h,b,raw,sha

def prepare(folder,fold,dates,y,h,b,raw,sha):
    from reconstruction import rich_features
    from interpretable_models import compact_features
    month=np.array([str(d)[:7] for d in dates]);day=np.array([int(str(d)[8:10]) for d in dates])
    train=(month>=fold['train_start'])&(month<=fold['train_end']);select=(month==fold['validation'])&(day<=15)
    cal=(month==fold['validation'])&(day>15);test=month==fold['test']
    ids=np.flatnonzero(train|select|cal|test);nt=int(train.sum());ns=int(select.sum());nc=int(cal.sum());ne=int(test.sum())
    assert max(dates[train])<min(dates[select])<=max(dates[select])<min(dates[cal])<=max(dates[cal])<min(dates[test])
    probe=np.flatnonzero(test)[0];changed=y.copy();changed[probe:]+=np.arange(4)*1234
    assert np.array_equal(raw['rich'][probe:probe+1].reshape(96,-1),rich_features(dates[probe:probe+1],h[probe:probe+1],dates,changed))
    assert np.array_equal(raw['compact'][probe:probe+1],compact_features(dates[probe:probe+1],h[probe:probe+1],dates,changed,True)[0])
    folder.mkdir(parents=True,exist_ok=True)
    for name,x in raw.items():
        mu=x[train].mean((0,1,2));sd=x[train].std((0,1,2)).clip(.05)
        xx=np.clip((x[ids]-mu)/sd,-15,15).astype('float32')
        if name=='compact':xx=np.concatenate([xx,np.broadcast_to(np.eye(4)[None,:,None,:],(len(ids),4,24,4))],-1).astype('float32')
        knots=np.quantile(xx[:nt].reshape(-1,xx.shape[-1]),[.25,.5,.75],axis=0).astype('float32')
        np.savez(folder/(name+'.npz'),x=xx,mu=mu,sd=sd,knots=knots)
    np.savez(folder/'data.npz',dates=dates[ids].astype('U10'),y=y[ids],h=h[ids],b=b[ids],train_end=nt,select_end=nt+ns,test_start=nt+ns+nc)
    meta=dict(**fold,train_days=nt,selection_days=ns,calibration_days=nc,test_days=ne,sha256=sha,
              boundaries=[str(dates[ids][i]) for i in [0,nt-1,nt,nt+ns-1,nt+ns,nt+ns+nc-1,nt+ns+nc,len(ids)-1]])
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2));return meta

def train(folder,smoke):
    env=dict(os.environ,SIMPLE_SEARCH_ROOT=str(folder))
    subprocess.run([sys.executable,'code/expanding_simple.py','--worker',str(folder)]+(['--smoke'] if smoke else []),env=env,check=True)

def worker(folder,smoke):
    import simple_search as s
    assert s.ROOT==folder
    jobs=s.configs()
    if smoke:jobs=[next(c for c in jobs if c['family']==m and c['feature']==f) for m in ['linear','additive','mlp','cnn'] for f in ['compact','rich']]
    rows=s.pool_run(jobs,8,2 if smoke else 240,True)
    d=np.load(folder/'data.npz');nt=int(d['train_end']);ns=int(d['select_end'])-nt;nc=int(d['test_start'])-nt-ns
    # Add ridge controls, using the same rich features and train-only normalization.
    x=np.load(folder/'rich.npz')['x'].astype('float64').reshape(len(d['y'])*96,-1)
    x=np.column_stack([np.ones(len(x)),x]);target=((d['y']-d['b'])/20).reshape(-1,4)
    gram=x[:nt*96].T@x[:nt*96];cross=x[:nt*96].T@target[:nt*96];betas={}
    for alpha in ([1000] if smoke else [1,10,100,1000,10000]):
        penalty=np.eye(x.shape[-1])*alpha;penalty[0,0]=0;beta=np.linalg.solve(gram+penalty,cross)
        rr=(x@beta).reshape(-1,4,24,4)[nt:];pp=d['b'][nt:]+20*(rr-rr.mean(-1,keepdims=True))
        cid='ridge_'+str(alpha);ff=folder/'trials'/cid;ff.mkdir(parents=True,exist_ok=True)
        np.save(ff/'predictions.npy',pp);np.save(ff/'beta.npy',beta)
        rows.append(dict(config=dict(id=cid,family='ridge',feature='rich',anchor='interpolation',alpha=alpha),parameters=beta.size,
                         validation_mae=float(np.abs(pp[:ns]-d['y'][nt:nt+ns]).mean())))
    ff=folder/'trials'/'interpolation';ff.mkdir(parents=True,exist_ok=True);np.save(ff/'predictions.npy',d['b'][nt:])
    rows.append(dict(config=dict(id='interpolation',family='interpolation'),parameters=0,validation_mae=float(np.abs(d['b'][nt:nt+ns]-d['y'][nt:nt+ns]).mean())))
    families=['linear','additive','mlp','cnn','ridge','interpolation']
    winners={m:min([r for r in rows if r['config']['family']==m],key=lambda r:r['validation_mae']) for m in families}
    # Persist selections before reading calibration/test targets for uncertainty or scoring.
    (folder/'selection.json').write_text(json.dumps(dict(by_family=winners,candidates=rows),indent=2))
    report=dict(manifest=json.loads((folder/'manifest.json').read_text()),selected={},candidates=rows)
    for family,row in winners.items():
        pred=np.load(folder/'trials'/str(row['config']['id'])/'predictions.npy');calpred=pred[ns:ns+nc];testpred=pred[ns+nc:]
        err=np.abs(testpred-d['y'][nt+ns+nc:]);calerr=np.abs(calpred-d['y'][nt+ns:nt+ns+nc])
        assert np.isfinite(pred).all() and np.abs(pred.mean(-1)-d['h'][nt:]).max()<1e-8
        intervals,bands=calibrate(calerr,err)
        rr=dict(row,test_mae=float(err.mean()),by_zone=err.mean((0,2,3)).tolist(),daily_mae=err.mean((1,2,3)).tolist(),intervals=intervals)
        report['selected'][family]=rr
        saved=dict(dates=d['dates'][nt+ns+nc:],observed=d['y'][nt+ns+nc:],predicted=testpred,calibration_error=calerr)
        for key,radius in bands.items():saved['lower_'+key]=testpred-radius;saved['upper_'+key]=testpred+radius
        np.savez_compressed(folder/(family+'_test.npz'),**saved)
        # Marginal intervals companion CSV; point forecasts remain challenge-compatible.
        frame=pd.DataFrame([(date,f'ZONE_{z}',hr+1,q+1,testpred[dd,z,hr,q]) for dd,date in enumerate(saved['dates']) for z in range(4) for hr in range(24) for q in range(4)],columns=['delivery_date','market_id','hour_index','quarter_index','predicted_price_eur_mwh'])
        for level in [50,90,95]:
            for side in ['lower','upper']:frame[f'{side}_{level}']=saved[f'{side}_marginal_{level}'].ravel()
        frame.to_csv(folder/(family+'_intervals.csv'),index=False)
    report['audits']='passed: strict split boundaries, source hash, causal features, train-only normalization/knots/gradient updates, frozen selection, finite conserved points, neural serialized replay, corrected conformal ranks'
    (folder/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))

def summarize(out,reports):
    rows=[]
    for r in reports:
        meta=r['manifest']
        for family,m in r['selected'].items():
            row=dict(**meta,family=family,parameters=m['parameters'],selection_mae=m['validation_mae'],test_mae=m['test_mae'])
            row.update({f'mae_ZONE_{z}':v for z,v in enumerate(m['by_zone'])})
            row.pop('boundaries');row.pop('sha256')
            for c in m['intervals']:
                if c['kind']=='marginal':row[f'coverage_{round(c["nominal"]*100)}']=c['coverage'];row[f'width_{round(c["nominal"]*100)}']=c['width']
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'learning_curves.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(dict(folds=reports,audits='all completed fold audits passed',caveat='Developmental resampling; calibration dates were inspected in earlier campaigns. Chronological split prevents reuse within this campaign. Time dependence/drift invalidate an unconditional nominal-coverage promise.'),indent=2,allow_nan=False))
    plot(out,frame)

def plot(out,frame):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for plan in frame.plan.unique():
        ff=frame[frame.plan==plan];fig,axes=plt.subplots(1,2,figsize=(13,4.7))
        for family,g in ff.groupby('family'):
            g=g.sort_values('train_days');axes[0].plot(g.train_days,g.test_mae,'o-',label=family)
            axes[1].plot(g.train_days,g.coverage_90*100,'o-',label=family)
        axes[1].axhline(90,color='gray',ls='--');axes[0].set(ylabel='Test MAE (EUR/MWh)',xlabel='Training days');axes[1].set(ylabel='Observed 90% marginal coverage (%)',xlabel='Training days',ylim=(0,102))
        for ax in axes:ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Expanding training; test months February–September' if plan=='expanding' else 'Fixed August selection/calibration and September test; add older history');fig.tight_layout()
        fig.savefig(out/(plan+'.png'),dpi=170);fig.savefig(out/(plan+'.svg'));plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for z,ax in enumerate(axes.flat):
            for family,g in ff.groupby('family'):
                g=g.sort_values('train_days');ax.plot(g.train_days,g[f'mae_ZONE_{z}'],'o-',label=family)
            ax.set(title=f'ZONE_{z}',xlabel='Training days',ylabel='Test MAE (EUR/MWh)');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle(plan.replace('_',' '));fig.tight_layout()
        fig.savefig(out/(plan+'_by_zone.png'),dpi=170);fig.savefig(out/(plan+'_by_zone.svg'));plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',type=Path);a=p.parse_args()
    self_test()
    if a.worker:worker(a.worker,a.smoke);return
    out=OUT if not a.smoke else Path(str(OUT)+'_smoke');out.mkdir(parents=True,exist_ok=True)
    dates,y,h,b,raw,sha=load_raw();jobs=folds(dates)
    if a.smoke:jobs=[jobs[0],jobs[8]]
    reports=[]
    for i,fold in enumerate(jobs):
        folder=out/(fold['plan']+'_'+str(fold['months'])+'m')
        prepare(folder,fold,dates,y,h,b,raw,sha);train(folder,a.smoke)
        reports.append(json.loads((folder/'metrics.json').read_text()));summarize(out,reports)
        (out/'progress.json').write_text(json.dumps(dict(completed_folds=i+1,total_folds=len(jobs),last=fold),indent=2))
        print('FOLD DONE',i+1,len(jobs),fold,flush=True)
    print('DONE expanding simple search with conformal calibration',flush=True)

if __name__=='__main__':main()
