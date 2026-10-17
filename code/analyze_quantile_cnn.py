"""Held-out quantile calibration, chronological error-risk probes, and CNN explanations."""
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from quantile_cnn import (ROOT, TAUS, TRAIN_END, SELECT_END, CAL_END, SEVERE_MAE,
                          normalize, make_features, load_model, write_json, configure)

INTERVALS = {'50': (4, 6), '80': (2, 8), '90': (1, 9), '95': (0, 10)}


def quantile_analysis():
    d = np.load(ROOT/'data.npz')
    selection = json.loads((ROOT/'selection.json').read_text())
    reports = {}
    for key, row in selection['by_group'].items():
        if not row['config']['quantile_weight']:
            continue
        folder = ROOT/'trials'/str(row['config']['id'])
        p = np.load(folder/'predictions.npz'); q = p['quantiles']
        truth = d['y'][TRAIN_END:]
        ca = slice(SELECT_END-TRAIN_END, CAL_END-TRAIN_END)
        te = slice(CAL_END-TRAIN_END, None)
        err = truth[te, ..., None] - q[te]
        pinball = np.maximum(TAUS*err, (TAUS-1)*err).mean((0,1,2,3))
        coverage = (truth[te, ..., None] <= q[te]).mean((0,1,2,3))
        intervals, saved = {}, {}
        for level, (lo, hi) in INTERVALS.items():
            alpha = 1-int(level)/100
            scores = np.maximum(np.maximum(q[ca,...,lo]-truth[ca],truth[ca]-q[ca,...,hi]),0)
            # Zone/quarter-position groups, pooling the16days x24hours. Dependence is reported.
            n = scores.shape[0]*scores.shape[2]
            rank = int(np.ceil((n+1)*(1-alpha)))-1
            assert rank < n
            offsets = np.sort(scores.transpose(1,3,0,2).reshape(4,4,-1),axis=-1)[...,rank]
            lower, upper = q[te,...,lo], q[te,...,hi]
            cal_lower, cal_upper = lower-offsets[None,:,None,:], upper+offsets[None,:,None,:]
            raw_hits = (truth[te]>=lower)&(truth[te]<=upper)
            cal_hits = (truth[te]>=cal_lower)&(truth[te]<=cal_upper)
            intervals[level] = dict(raw_coverage=float(raw_hits.mean()),raw_width=float((upper-lower).mean()),
                calibrated_coverage=float(cal_hits.mean()),calibrated_width=float((cal_upper-cal_lower).mean()),
                by_zone_coverage=cal_hits.mean((0,2,3)).tolist(),
                whole_day_coverage=float(cal_hits.all((1,2,3)).mean()),
                offsets=offsets.tolist(),calibration_points_per_group=n,corrected_order_statistic=rank+1)
            saved[f'lower_{level}'],saved[f'upper_{level}']=cal_lower,cal_upper
        np.savez(folder/'calibrated_intervals.npz',**saved)
        reports[key] = dict(taus=TAUS.tolist(),pinball_by_tau=pinball.tolist(),quantile_coverage=coverage.tolist(),
                           intervals=intervals,crossing_rate=float((np.diff(q,axis=-1)<-1e-7).mean()),
                           point_vs_q50_mae=float(np.abs(p['point'][te]-q[te,...,5]).mean()))
    write_json(ROOT/'quantile_metrics.json',dict(by_group=reports,
        caveat='Calibration uses August16-31 only. Pointwise empirical coverage under temporal dependence; no exchangeability or whole-day guarantee.'))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for key,r in reports.items():
        axes[0].plot(TAUS,r['quantile_coverage'],'o-',label=key)
        levels=[int(k)/100 for k in INTERVALS]
        axes[1].plot(levels,[r['intervals'][k]['calibrated_coverage'] for k in INTERVALS],'o-',label=key+' calibrated')
        axes[1].plot(levels,[r['intervals'][k]['raw_coverage'] for k in INTERVALS],'x--',label=key+' raw')
    for ax in axes:
        ax.plot([0,1],[0,1],'k:',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Nominal probability',ylabel='September empirical fraction');ax.legend(fontsize=8)
    axes[0].set_title('All eleven marginal quantiles');axes[1].set_title('Pointwise intervals; separate August calibration')
    fig.savefig(ROOT/'quantile_calibration.png',dpi=150);plt.close(fig)


def risk_features(d, indices, pred):
    """Only supplied hourly prices, strictly prior observations and model outputs."""
    rows=[]
    names=['hourly_mean','hourly_std','hourly_min','hourly_max','mean_abs_hour_ramp','max_abs_hour_ramp',
           'mean_zone_spread','max_zone_spread','history_within_hour_abs','history_within_hour_max','history_fraction',
           'width50_mean','width90_mean','width90_max','width95_mean','point_q50_gap_mean','point_q50_gap_max',
           'point_correction_mean','point_correction_max',
           'conv_mean','conv_std','conv_max','conv_active_fraction',
           'fusion_mean','fusion_std','fusion_max','fusion_active_fraction']
    for j,idx in enumerate(indices):
        h=d['h'][idx];day=dt.date.fromisoformat(str(d['dates'][idx]))
        lower=str(day-dt.timedelta(days=7));date=str(day)
        prior=(d['dates']>=lower)&(d['dates']<date)
        hist=d['y'][prior];res=hist-hist.mean(-1,keepdims=True) if len(hist) else np.zeros((1,4,24,4))
        ramp=np.abs(np.diff(h,axis=-1));spread=np.ptp(h,axis=0)
        q,p=pred['quantiles'][j],pred['point'][j]
        gap=np.abs(p-q[...,5]);corr=np.abs(p-d['b'][idx])
        row=[h.mean(),h.std(),h.min(),h.max(),ramp.mean(),ramp.max(),spread.mean(),spread.max(),
             np.abs(res).mean(),np.abs(res).max(),len(hist)/7,
             (q[...,6]-q[...,4]).mean(),(q[...,9]-q[...,1]).mean(),(q[...,9]-q[...,1]).max(),
             (q[...,10]-q[...,0]).mean(),gap.mean(),gap.max(),corr.mean(),corr.max()]
        for layer in ['conv','fused']:
            a=pred[layer][j];row += [a.mean(),a.std(),a.max(),(a>0).mean()]
        rows.append(row)
    return np.asarray(rows,dtype=float),names


def fit_logistic(x,y):
    mu=x.mean(0);sd=x.std(0).clip(.05);z=np.clip((x-mu)/sd,-15,15)
    if min(int(y.sum()),int((1-y).sum()))<3:
        p=(y.sum()+1)/(len(y)+2)
        return dict(mu=mu.tolist(),sd=sd.tolist(),weight=np.zeros(x.shape[1]).tolist(),bias=float(np.log(p/(1-p))),
                    status='constant fallback: fewer than3 examples in one class')
    def loss(theta):
        logits=z@theta[:-1]+theta[-1]
        value=np.mean(np.logaddexp(0,logits)-y*logits)+.05*np.dot(theta[:-1],theta[:-1])
        err=expit(logits)-y
        grad=np.r_[z.T@err/len(y)+.1*theta[:-1],err.mean()]
        return value,grad
    result=minimize(loss,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
    assert result.success and np.isfinite(result.x).all()
    return dict(mu=mu.tolist(),sd=sd.tolist(),weight=result.x[:-1].tolist(),bias=float(result.x[-1]),status='fitted')


def logits(model,x):
    z=np.clip((x-np.asarray(model['mu']))/np.asarray(model['sd']),-15,15)
    return z@np.asarray(model['weight'])+model['bias']


def probability_metrics(y,p,threshold):
    labels=y.astype(bool);n=len(y);positive=int(labels.sum())
    order=np.argsort(-p,kind='stable');top=order[:max(1,int(np.ceil(.1*n)))]
    flags=p>threshold
    constant=bool(np.ptp(p)<1e-12)
    ap=(float(y.mean()) if constant else float(np.mean(np.cumsum(labels[order])[labels[order]]/
          np.arange(1,n+1)[labels[order]]))) if positive else None
    auc=float(((p[labels,None]>p[None,~labels])+.5*(p[labels,None]==p[None,~labels])).mean()) if 0<positive<n else None
    bins=[]
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if mask.any():bins.append(dict(count=int(mask.sum()),predicted=float(p[mask].mean()),observed=float(y[mask].mean())))
    return dict(days=n,events=positive,prevalence=float(y.mean()),brier=float(np.mean((p-y)**2)),average_precision=ap,auc=auc,
                fixed_threshold=float(threshold),alert_fraction=float(flags.mean()),
                alert_precision=float(y[flags].mean()) if flags.any() else None,
                alert_recall=float(labels[flags].sum()/positive) if positive else None,
                top10percent_recall=float(labels[top].sum()/positive) if positive and not constant else None,
                reliability=bins,constant_score=constant)


def risk_analysis():
    d=np.load(ROOT/'data.npz');xs=[];ys=[];errs=[];inds=[]
    manifest=json.loads((ROOT/'oof_manifest.json').read_text())
    for fold in manifest['folds']:
        p=np.load(ROOT/'oof'/str(fold['fold'])/'predictions.npz')
        ix=np.arange(int(p['start']),int(p['end']))
        x,names=risk_features(d,ix,p)
        error=np.abs(p['point']-d['y'][ix]).mean((1,2,3))
        xs.append(x);ys.append((error>SEVERE_MAE).astype(float));errs.append(error);inds.append(ix)
    x,y,error,indices=np.concatenate(xs),np.concatenate(ys),np.concatenate(errs),np.concatenate(inds)
    selection=json.loads((ROOT/'selection.json').read_text())
    folder=ROOT/'trials'/str(selection['quantile']['config']['id'])
    p=np.load(folder/'predictions.npz');ix=np.arange(TRAIN_END,len(d['dates']))
    future,checknames=risk_features(d,ix,p);assert names==checknames
    future_error=np.abs(p['point']-d['y'][ix]).mean((1,2,3));fy=(future_error>SEVERE_MAE).astype(float)
    ca=slice(SELECT_END-TRAIN_END,CAL_END-TRAIN_END);te=slice(CAL_END-TRAIN_END,None)
    models={};scores={}
    for name,count in [('inputs',11),('inputs_uncertainty',19),('plus_conv',23),('plus_fusion',27)]:
        model=fit_logistic(x[:,:count],y)
        model['feature_names']=names[:count]
        cal_logits=logits(model,future[ca,:count])
        # One intercept adjustment on a separate16-day block, regularized toward zero.
        def loss(offset):return float(np.sum(np.logaddexp(0,cal_logits+offset)-fy[ca]*(cal_logits+offset))+.5*offset**2)
        offset=float(minimize_scalar(loss,bounds=(-4,4),method='bounded').x)
        probs=expit(logits(model,future[:,:count])+offset)
        threshold=float(np.quantile(probs[ca],.9))
        model.update(calibration_intercept=offset,alert_threshold=threshold,
                     september=probability_metrics(fy[te],probs[te],threshold),
                     calibration=probability_metrics(fy[ca],probs[ca],threshold))
        models[name]=model;scores[name]=probs
    np.savez(ROOT/'risk_predictions.npz',dates=d['dates'][ix],indices=ix,features=future,actual_mae=future_error,
             severe=fy,**scores)
    np.savez(ROOT/'risk_oof.npz',indices=indices,dates=d['dates'][indices],features=x,severe=y,actual_mae=error)
    write_json(ROOT/'risk_models.json',dict(threshold_eur_mwh=SEVERE_MAE,features=names,models=models,
        training_days=len(y),training_events=int(y.sum()),
        caveat='Risk fit on chronological out-of-sample errors; activation summaries are channel-permutation invariant, not neuron identities. Calibration has16days only; probabilities exploratory. All probes reported without September selection. Top10% recall is a batch ranking diagnostic; deployable threshold fixed on calibration.'))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,m in models.items():
        bins=m['september']['reliability']
        axes[0].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=name)
    axes[0].plot([0,1],[0,1],'k:');axes[0].set(xlabel='Predicted severe-day probability',ylabel='Observed severe-day fraction',title='September risk reliability (30 days)');axes[0].legend(fontsize=8)
    score=scores['plus_fusion'][te];err=future_error[te];order=np.argsort(score)
    axes[1].scatter(score,err,c=np.arange(len(err)),cmap='viridis');axes[1].axhline(SEVERE_MAE,color='r',ls=':')
    axes[1].set(xlabel='Predicted risk including layer summaries',ylabel='Actual day MAE (EUR/MWh)',title='Does risk identify large errors?')
    fig.savefig(ROOT/'day_risk.png',dpi=150);plt.close(fig)


def cam(net,x,zone,hour,quarter,target):
    point,q,conv,_=net(x);conv.retain_grad()
    scalar=point[0,zone,hour,quarter] if target=='point' else q[0,zone,hour,quarter,9]-q[0,zone,hour,quarter,1]
    net.zero_grad();(20*scalar).backward()
    activation=conv.detach().cpu().numpy()[0];gradient=conv.grad.detach().cpu().numpy()[0]
    signed=(gradient.mean((1,2))[:,None,None]*activation).sum(0)
    local=(gradient*activation).sum(0)
    return dict(activation=activation,gradient=gradient,gradcam_signed=signed,gradcam_positive=np.maximum(signed,0),local_signed=local)


def explanations():
    configure();d=np.load(ROOT/'data.npz');selection=json.loads((ROOT/'selection.json').read_text())
    folder=ROOT/'trials'/str(selection['quantile']['config']['id']);net,ck=load_model(folder)
    f=np.load(ROOT/f"{ck['config']['pack']}.npz");p=np.load(folder/'predictions.npz')
    rp=np.load(ROOT/'risk_predictions.npz');errors=rp['actual_mae'][CAL_END-TRAIN_END:]
    cases=sorted(set([CAL_END,CAL_END+int(errors.argmax()),CAL_END+int(rp['plus_fusion'][CAL_END-TRAIN_END:].argmax())]))
    outdir=ROOT/'explanations';outdir.mkdir(exist_ok=True);summary=[];ablations=[]
    weights=net.conv.weight.detach().cpu().numpy()
    np.savez(outdir/'learned_filters.npz',conv=weights,fusion=net.fusion.weight.detach().cpu().numpy(),
             point=net.point.weight.detach().cpu().numpy(),median=net.median.weight.detach().cpu().numpy(),
             increments=net.increments.weight.detach().cpu().numpy(),mu=ck['mu'],sd=ck['sd'])
    for ch in range(weights.shape[1]):
        fig,axes=plt.subplots(4,4,figsize=(11,6),layout='constrained');lim=max(float(np.abs(weights[:,ch]).max()),1e-8)
        for k,ax in enumerate(axes.flat):
            ax.imshow(weights[k,ch],aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.set(title=f'Filter{k}',yticks=[0,1],yticklabels=['Q','H'],xticks=[])
        fig.suptitle(f'Input channel{ch}; weights for normalized inputs')
        fig.savefig(outdir/f'kernels_channel_{ch}.png',dpi=120);plt.close(fig)
    for idx in cases:
        date=str(d['dates'][idx]);j=idx-TRAIN_END
        x=torch.tensor(normalize(f['raw'][idx:idx+1],ck['mu'],ck['sd'],ck['config']['pack']),device='cuda',requires_grad=True)
        correction=np.abs(p['point'][j]-d['b'][idx]);zone,hour,quarter=np.unravel_index(correction.argmax(),correction.shape)
        maps={}
        for target in ['point','width90']:
            result=cam(net,x,zone,hour,quarter,target);maps.update({target+'_'+k:v for k,v in result.items()})
            fig,axes=plt.subplots(2,1,figsize=(12,5),layout='constrained')
            for ax,name in zip(axes,['gradcam_signed','local_signed']):
                values=result[name];lim=max(float(np.abs(values).max()),1e-8)
                im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
                ax.set(yticks=range(8),yticklabels=[f'lag{k}' for k in range(7,-1,-1)],xlabel='Quarter-hour position',title=name)
                fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle(f'{date}: zone{zone}, hour{hour+1}, quarter{quarter+1}; {target}')
            fig.savefig(outdir/f'{date}_{target}.png',dpi=140);plt.close(fig)
        # Empirical model-randomization sanity check, reported without a fabricated pass criterion.
        randomized,_=load_model(folder);torch.manual_seed(123)
        with torch.no_grad():
            for parameter in randomized.parameters():parameter.normal_(0,.1)
        random_map=cam(randomized,x,zone,hour,quarter,'point')['local_signed']
        normal=maps['point_local_signed'];den=normal.std()*random_map.std()
        correlation=float(np.corrcoef(normal.ravel(),random_map.ravel())[0,1]) if den>1e-12 else None
        np.savez(outdir/f'{date}_maps.npz',**maps,randomized_local_signed=random_map)
        day=dt.date.fromisoformat(date)
        for lag in list(range(1,8))+['all7']:
            removed={str(day-dt.timedelta(days=k)) for k in (range(1,8) if lag=='all7' else [lag])}
            keep=np.array([str(a) not in removed for a in d['dates']])
            raw=make_features(d['dates'][idx:idx+1],d['h'][idx:idx+1],d['dates'][keep],d['y'][keep],ck['config']['pack'])
            xx=torch.tensor(normalize(raw,ck['mu'],ck['sd'],ck['config']['pack']),device='cuda')
            with torch.no_grad():point,q,_,_=net(xx)
            pred=d['b'][idx]+20*point.cpu().numpy()[0].astype(float)
            ablations.append(dict(date=date,removed_calendar_lag=lag,point_change=float(np.abs(pred-p['point'][j]).mean()),
                                  day_mae=float(np.abs(pred-d['y'][idx]).mean())))
        summary.append(dict(date=date,zone=int(zone),hour=int(hour+1),quarter=int(quarter+1),
                            day_mae=float(rp['actual_mae'][j]),randomization_local_map_correlation=correlation))
    write_json(outdir/'summary.json',dict(cases=summary,ablations=ablations,
        caveat='Posthoc diagnostic examples: first day, worst actual MAE, highest predicted risk. Explanations do not select models. Masks and analog retrieval recomputed consistently for ablations. Grad-CAM targets point correction and90% width, never a fixed day sum.'))


def report():
    m=json.loads((ROOT/'metrics.json').read_text());q=json.loads((ROOT/'quantile_metrics.json').read_text());risk=json.loads((ROOT/'risk_models.json').read_text())
    lines=['# Small 2D CNN with quantiles and day-error risk','',
           'Selection: August1–15. Calibration: August16–31. September is developmental.','',
           f"August-selected point model: **{m['selected_group']}**. Quantile model: **{m['quantile_group']}**.",'',
           '| Input/loss | Point parameters | Total parameters | Selection MAE | September MAE |',
           '|---|---:|---:|---:|---:|']
    for name,r in m['by_group'].items():lines.append(f"| {name} | {r['point_parameters']} | {r['parameters']} | {r['validation_mae']:.6f} | {r['september_mae']:.6f} |")
    lines += ['', 'The practical target is approximately5.2 MAE; no September-based search or stopping. Only point predictions conserve hourly means.', '',
              '| Quantile model | Raw90% coverage | Calibrated90% coverage | Calibrated width |', '|---|---:|---:|---:|']
    for name,r in q['by_group'].items():
        v=r['intervals']['90'];lines.append(f"| {name} | {v['raw_coverage']:.3f} | {v['calibrated_coverage']:.3f} | {v['calibrated_width']:.3f} |")
    lines += ['',f"Severe day: joint four-zone day MAE > {SEVERE_MAE} EUR/MWh. Risk labels: {risk['training_days']} held-out earlier days, {risk['training_events']} events.",'',
              '| Risk features | September events | Brier | Average precision | Recall among top10% |', '|---|---:|---:|---:|---:|']
    for name,r in risk['models'].items():
        s=r['september'];ap='n/a' if s['average_precision'] is None else f"{s['average_precision']:.3f}"
        rec='n/a' if s['top10percent_recall'] is None else f"{s['top10percent_recall']:.3f}"
        lines.append(f"| {name} | {s['events']} | {s['brier']:.3f} | {ap} | {rec} |")
    lines += ['',risk['caveat'],'',q['caveat'],'','See explanations/ for normalized kernels, signed maps, randomization checks, and consistent history ablations.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');names=list(m['by_group']);values=[m['by_group'][k]['september_mae'] for k in names]
    ax.bar(names,values);ax.axhline(5.2,color='r',ls=':',label='Practical target5.2')
    ax.set(ylabel='September MAE (EUR/MWh)',title='Matched inputs and losses; August-selected models');ax.legend()
    fig.savefig(ROOT/'mae_comparison.png',dpi=150);plt.close(fig)
    print((ROOT/'REPORT.md').read_text(),flush=True)


if __name__=='__main__':
    quantile_analysis();risk_analysis();explanations();report()
