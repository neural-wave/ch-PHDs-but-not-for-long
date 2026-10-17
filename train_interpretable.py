"""Remote compact linear/GAM comparison with inspectable coefficients, no ensemble."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from reconstruction import read_targets,baseline,ZONES
from hierarchical_models import conformal_quantile
from interpretable_models import BASIS,compact_features,design,infer

out=Path('results/interpretable_vast');out.mkdir(parents=True,exist_ok=True)
dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);b=baseline(h)
assert len(dates)==363 and dates[302]=='2026-08-01' and dates[317]=='2026-08-16' and dates[333]=='2026-09-01'
assert np.max(np.abs(BASIS@BASIS.T-np.eye(3)))<1e-12 and np.max(np.abs(BASIS.sum(-1)))<1e-12
reports=[];bundles=[];predictions=[];max_equivalence=0
for history in [False,True]:
    x,names=compact_features(dates,h,dates,y,history)
    mutated=y.copy();mutated[333:]+=np.arange(4)*1000
    assert np.array_equal(x[333:334],compact_features(dates[333:334],h[333:334],dates,mutated,history)[0])
    for family in ['linear','additive']:
        for anchor_name in ['flat','interpolation']:
            anchor=b if anchor_name=='interpolation' else np.broadcast_to(h[...,None],y.shape)
            target=(y-anchor)/20
            c=target@BASIS.T
            for alpha in [1,10,100,1000]:
                start=time.perf_counter();states=[];cs=[];direct=[]
                for z in range(4):
                    xx=x[:,z];mu=xx[:302].mean((0,1));sd=xx[:302].std((0,1)).clip(.05)
                    knots=np.quantile(((xx[:302]-mu)/sd).reshape(-1,len(names)),[.25,.5,.75],axis=0) if family=='additive' else None
                    dd=design(xx,mu,sd,knots);train=dd[:302].reshape(-1,dd.shape[-1])
                    penalty=np.eye(dd.shape[-1])*alpha;penalty[0,0]=0
                    gram=train.T@train+penalty
                    beta=np.linalg.solve(gram,train.T@c[:302,z].reshape(-1,3))
                    # Matched direct-quarter ridge has exactly the same function class/loss.
                    dbeta=np.linalg.solve(gram,train.T@target[:302,z].reshape(-1,4))
                    states.append(dict(mu=mu,sd=sd,knots=knots,beta=beta))
                    cs.append(dd@beta);direct.append(dd@dbeta)
                pp=anchor+20*np.stack(cs,axis=1)@BASIS
                direct=anchor+20*np.stack(direct,axis=1)
                delta=float(np.max(np.abs(pp-direct)));max_equivalence=max(max_equivalence,delta)
                assert delta<1e-8
                assert np.isfinite(pp).all() and np.abs(pp.mean(-1)-h).max()<1e-8
                bundles.append(dict(family=family,anchor=anchor_name,history=history,names=names,zones=states,alpha=alpha))
                predictions.append(pp)
                reports.append(dict(name=f'{family}_{anchor_name}_history{int(history)}_alpha{alpha}',family=family,anchor=anchor_name,
                    history=history,alpha=alpha,raw_features=len(names),parameters=sum(s['beta'].size for s in states),
                    selection_mae=float(np.abs(pp[302:317]-y[302:317]).mean()),seconds=time.perf_counter()-start))
                (out/'progress.json').write_text(json.dumps(reports,indent=2))
                print(reports[-1],flush=True)
selected=int(np.argmin([m['selection_mae'] for m in reports]));bundle=bundles[selected];pred=predictions[selected]
# Freeze selected index before calculating any calibration/September metrics.
for meta,pp in zip(reports,predictions):
    meta.update(august_mae=float(np.abs(pp[302:333]-y[302:333]).mean()),september_mae=float(np.abs(pp[333:]-y[333:]).mean()))
serial=dict(bundle)
serial['zones']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items()} for s in bundle['zones']]
(out/'model.json').write_text(json.dumps(serial,indent=2))
loaded=json.loads((out/'model.json').read_text())
for state in loaded['zones']:
    for key in ['mu','sd','beta','knots']:
        if state[key] is not None: state[key]=np.asarray(state[key])
replay=infer(loaded,dates[333:],h[333:],dates,y)
assert np.max(np.abs(replay-pred[333:]))<1e-10
cal=np.abs(pred[317:333]-y[317:333]);err=np.abs(pred[333:]-y[333:]);bands={};conformal=[]
for level in [.5,.9,.95]:
    radius=conformal_quantile(cal.transpose(0,2,1,3).reshape(-1,4,4),1-level)[None,:,None,:]
    covered=err<=radius
    conformal.append(dict(nominal=level,coverage=float(covered.mean()),mean_width=float(2*radius.mean()),radius=radius.ravel().tolist()))
    bands[f'lower_{round(100*level)}']=pred[333:]-radius;bands[f'upper_{round(100*level)}']=pred[333:]+radius
np.savez_compressed(out/'test_predictions.npz',dates=dates[333:].astype('U10'),observed=y[333:],predicted=pred[333:],**bands)
coef=[]
for z,state in enumerate(bundle['zones']):
    names=['intercept']+bundle['names']
    if state['knots'] is not None:
        names += [f'{name}_hinge_q{q}' for q in [25,50,75] for name in bundle['names']]
    for name,values in zip(names,state['beta']):
        for contrast,value in zip(['half_hour','first_quarter_pair','second_quarter_pair'],values):
            coef.append(dict(zone=ZONES[z],feature=name,contrast=contrast,standardized_coefficient=float(value)))
pd.DataFrame(coef).to_csv(out/'coefficients.csv',index=False)
rows=[(dates[d],zone,hr+1,q+1,pred[d,z,hr,q]) for d in range(333,363) for z,zone in enumerate(ZONES) for hr in range(24) for q in range(4)]
frame=pd.DataFrame(rows,columns=['delivery_date','market_id','hour_index','quarter_index','predicted_price_eur_mwh'])
frame.to_csv(out/'test_predictions.csv',index=False)
csv=pd.read_csv(out/'test_predictions.csv')
assert len(csv)==11520 and not csv.duplicated(list(frame.columns[:4])).any()
assert np.abs(csv.predicted_price_eur_mwh.to_numpy().reshape(30,4,24,4).mean(-1)-h[333:]).max()<1e-8
report=dict(source_sha256=hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest(),train_days=302,
    selection_days=15,calibration_days=16,test_days=30,selected=reports[selected],candidates=reports,conformal=conformal,
    hierarchy_direct_linear_max_difference=max_equivalence,serialized_replay_max_difference=float(np.max(np.abs(replay-pred[333:]))),
    max_hourly_mean_error=float(np.abs(pred.mean(-1)-h).max()),csv_audit='passed:11520 finite unique rows/exact means',
    caveat='Squared-loss linear/additive fitting; absolute-error validation selection. Orthogonal hierarchical and direct parameterizations are equivalent under this matched ridge objective. September is developmental; coverage empirical under dependence.')
(out/'metrics.json').write_text(json.dumps(report,indent=2))
print('DONE',json.dumps(report['selected']),flush=True)
