"""Independent artifact audit for ordered quantiles, calibration and confidence."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train as e
from audit_selection import audit_selection
import numpy as np
import pandas as pd
import torch


def close(a,b,tol=1e-8):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol)


def audit(smoke=False):
    e.p.setup();root=e.ROOT/'smoke' if smoke else e.ROOT
    manifest=json.loads((e.ROOT/'manifest.json').read_text())
    for path,sha in manifest['code_hashes'].items():assert hashlib.sha256((e.PROJECT/path).read_bytes()).hexdigest()==sha,path
    assert hashlib.sha256((e.PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==e.p.SHA
    assert hashlib.sha256((e.ROOT/'raw.npz').read_bytes()).hexdigest()==manifest['feature_source_sha256']
    dates,y=e.p.read_targets(e.PROJECT/'train_prices_15min.csv');h=y.mean(-1)
    uncorrected=np.asarray([[np.interp(np.arange(96)/4+.125,np.arange(24)+.5,zone).reshape(24,4) for zone in day] for day in h])
    b=uncorrected-uncorrected.mean(-1,keepdims=True)+h[...,None]
    raw=np.load(e.ROOT/'raw.npz');np.testing.assert_array_equal(raw['y'],y);close(raw['b'],b);np.testing.assert_array_equal(raw['dates'],dates)
    month=np.asarray([str(d)[:7] for d in dates]);daynums=np.asarray([int(str(d)[8:]) for d in dates])
    folders=sorted(path for path in (root if smoke else root/'folds').iterdir() if path.is_dir())
    assert len(folders)==(2 if smoke else 8)
    candidate_count=replayed=quarter_csv_rows=0;maximum_mean_error=maximum_replay_error=0.;all_monthly=[]
    for folder in folders:
        d=np.load(folder/'data.npz');meta=json.loads((folder/'manifest.json').read_text())
        nt,ns,nc,ne=(int(d[k]) for k in ['nt','ns','nc','ne'])
        assert ns==15 and meta['train_end']<meta['validation']<meta['test']
        train=(month>=meta['train_start'])&(month<=meta['train_end']);select=(month==meta['validation'])&(daynums<=15)
        cal=(month==meta['validation'])&(daynums>15);test=month==meta['test'];ids=np.flatnonzero(train|select|cal|test)
        assert [int(v.sum()) for v in [train,select,cal,test]]==[nt,ns,nc,ne]
        np.testing.assert_array_equal(d['y'],y[ids]);close(d['b'],b[ids]);np.testing.assert_array_equal(d['dates'],raw['dates'][ids])
        fitting_months=month[train];weights=np.zeros(nt)
        for m in sorted(set(fitting_months)):
            positions=fitting_months==m;denom=float(np.abs(y[train&(month==m)]-b[train&(month==m)]).mean())
            weights[positions]=1/(positions.sum()*denom)
        weights/=weights.mean();close(weights,d['weights_ratio'],2e-7);close(d['weights_uniform'],np.ones(nt))
        for feature in ['rich','compact','cnn','curve']:
            values=raw[feature];axes=tuple(range(values.ndim-1)) if feature in ['rich','compact'] else (0,2) if feature=='cnn' else (0,)
            mu=np.zeros_like(d[feature+'_mu']);sd=np.ones_like(d[feature+'_sd'])
            close(mu,d[feature+'_mu']);close(sd,d[feature+'_sd'])
            expected=np.clip((values[ids]-mu)/sd,-15,15).astype('float32')
            if feature=='compact':expected=np.concatenate([expected,np.broadcast_to(np.eye(4)[None,:,None,:],(len(ids),4,24,4))],-1).astype('float32')
            np.testing.assert_array_equal(expected,d[feature])
            if feature in ['rich','compact']:np.testing.assert_array_equal(np.quantile(expected[:nt].reshape(-1,expected.shape[-1]),[.25,.5,.75],axis=0).astype('float32'),d[feature+'_knots'])
        validation_baseline=float(np.abs(y[select]-b[select]).mean())
        selection,predictions,count,replays,violation,delta=audit_selection(e,folder,d,y[select],validation_baseline,smoke,close)
        winners=selection['by_recipe'];rows=selection['candidates']
        candidate_count+=count;replayed+=replays
        maximum_mean_error=max(maximum_mean_error,violation);maximum_replay_error=max(maximum_replay_error,delta)
        chosen=min((name for name in winners if name!='validation_selector'),key=lambda name:(winners[name]['selection_ratio'],name))
        assert winners['validation_selector']['chosen_recipe']==chosen
        assert (folder/'selection.json').stat().st_mtime_ns <= (folder/'metrics.json').stat().st_mtime_ns
        report=json.loads((folder/'metrics.json').read_text())
        for recipe,row in winners.items():
            allq=predictions[str(row['config']['id'])];calq=allq[ns:ns+nc];q=allq[ns+nc:]
            data=np.load(folder/'selected'/recipe/'predictions.npz');receipt=json.loads((folder/'selected'/recipe/'calibration.json').read_text())
            close(data['quantiles'],q);np.testing.assert_array_equal(data['observed'],y[test]);np.testing.assert_array_equal(data['dates'],raw['dates'][test])
            err=np.abs(q[...,5]-y[test]);baseline=float(np.abs(b[test]-y[test]).mean());score=report['results'][recipe]
            close(err.mean(),score['mae']);close(err.mean()/baseline,score['ratio']);close(err.mean((0,2,3)),score['by_zone'])
            pb_error=y[test,...,None]-q;pb=np.maximum(e.TAUS*pb_error,(e.TAUS-1)*pb_error).mean((0,1,2,3))
            close(pb,score['pinball_by_quantile']);close(pb.mean(),score['mean_pinball']);close(pb.mean()/baseline,score['pinball_ratio'])
            prior_lo=prior_hi=prior_cal_lo=prior_cal_hi=None
            for nominal,lo,hi in e.INTERVALS:
                level=round(100*nominal);scores=np.maximum(np.maximum(calq[...,lo]-y[cal],y[cal]-calq[...,hi]),0)
                pooled=scores.transpose(0,2,1,3).reshape(-1,4,4);rank=math.ceil((len(pooled)+1)*nominal)
                correction=np.sort(pooled,axis=0)[rank-1][None,:,None,:]
                assert receipt[str(level)]['rank']==rank and receipt[str(level)]['n']==len(pooled)
                close(receipt[str(level)]['correction'],correction[0,:,0,:])
                lower=q[...,lo]-correction;upper=q[...,hi]+correction
                cal_lower=calq[...,lo]-correction;cal_upper=calq[...,hi]+correction
                if prior_lo is not None:
                    lower=np.minimum(lower,prior_lo);upper=np.maximum(upper,prior_hi)
                    cal_lower=np.minimum(cal_lower,prior_cal_lo);cal_upper=np.maximum(cal_upper,prior_cal_hi)
                close(data[f'lower_{level}'],lower);close(data[f'upper_{level}'],upper)
                for kind,l,u in [('raw',q[...,lo],q[...,hi]),('calibrated',lower,upper)]:
                    r=next(v for v in score['intervals'] if v['kind']==kind and v['nominal']==nominal);covered=(y[test]>=l)&(y[test]<=u)
                    close(covered.mean(),r['coverage']);close((u-l).mean(),r['width']);close(covered.mean((0,2,3)),r['by_zone'])
                    close(covered.all(-1).mean(),r['whole_hour_coverage']);close(covered.all((1,2,3)).mean(),r['whole_day_coverage'])
                if level==90:cal_width90=cal_upper-cal_lower
                prior_lo,prior_hi,prior_cal_lo,prior_cal_hi=lower,upper,cal_lower,cal_upper
            width=data['upper_90']-data['lower_90'];percent=np.empty_like(width)
            for z,quarter in itertools.product(range(4),range(4)):
                reference=np.sort(cal_width90[:,z,:,quarter].ravel());values=width[:,z,:,quarter]
                percent[:,z,:,quarter]=(np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right'))/(2*len(reference))
            close(percent,data['uncertainty_percentile']);labels=np.where(percent<=.25,0,np.where(percent>=.75,2,1)).astype('int8')
            np.testing.assert_array_equal(labels,data['confidence_bin'])
            for index,r in enumerate(score['confidence_bins']):
                mask=labels==index;assert r['n']==int(mask.sum())
                if r['n']:
                    close(err[mask].mean(),r['mae']);close(np.abs(b[test]-y[test])[mask].mean(),r['interpolation_mae'])
                    close(width[mask].mean(),r['width90']);close(((y[test]>=data['lower_90'])&(y[test]<=data['upper_90']))[mask].mean(),r['coverage90'])
            all_monthly.append(dict(month=meta['test'],recipe=recipe,days=ne,mae=float(err.mean()),ratio=float(err.mean()/baseline),pinball_ratio=float(pb.mean()/baseline)))
        exported=pd.read_csv(folder/'predictions.csv.gz');chosen_data=np.load(folder/'selected'/'validation_selector'/'predictions.npz')
        assert len(exported)==ne*384 and not exported.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
        for k,tau in enumerate(e.TAUS):close(exported[f'q{tau:g}'],chosen_data['quantiles'][...,k].ravel(),1e-7)
        close(exported['width_90'],(chosen_data['upper_90']-chosen_data['lower_90']).ravel(),1e-7)
        close(exported['uncertainty_percentile'],chosen_data['uncertainty_percentile'].ravel(),1e-7)
        np.testing.assert_array_equal(exported['confidence_bin'],np.array(['narrow','middle','wide'])[chosen_data['confidence_bin'].ravel()])
        quarter_csv_rows+=len(exported)
    actual=pd.DataFrame(all_monthly);monthly=pd.read_csv(root/'monthly_scores.csv');assert len(actual)==len(monthly) and not monthly.duplicated(['month','recipe']).any()
    merged=monthly.merge(actual,on=['month','recipe'],validate='one_to_one',suffixes=('_saved','_audit'))
    for key in ['mae','ratio','pinball_ratio']:close(merged[key+'_saved'],merged[key+'_audit'])
    summary=pd.read_csv(root/'summary.csv')
    for row in summary.to_dict('records'):
        group=actual[actual.recipe==row['recipe']];close(group.ratio.mean(),row['mean_monthly_ratio']);close(np.average(group.mae,weights=group.days),row['pooled_mae'])
        close(group.pinball_ratio.mean(),row['mean_monthly_pinball_ratio']);close(group.ratio.max(),row['worst_ratio'])
    if not smoke:
        exported=pd.read_csv(root/'predictions.csv.gz');assert len(exported)==92544 and not exported.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
    result=dict(passed=True,smoke=smoke,folds=len(folders),candidate_predictions_recomputed=candidate_count,
                selected_neural_models_replayed=replayed,max_replay_error=maximum_replay_error,max_hourly_mean_error=maximum_mean_error,
                exported_quarter_rows=quarter_csv_rows,quantile_levels=e.TAUS.tolist(),
                checks='official/source/feature hashes, train-only weights/normalization/knots, quantile order/nonzero widths, conserved q50, validation argmins, quantile blends/selector, CQR ranks/corrections/nesting/coverage/widths, confidence percentiles/bins, CSV identity, checkpoint replay, point/pinball/aggregate scores')
    e.p.write(root/'audit.json',result)
    if not smoke:
        lines=['# Expanding-training optimization with fixed physical feature scaling','',
               'Every learned recipe predicts eleven ordered quantiles. The constrained q50 conserves supplied hourly means and determines MAE/interpolation_MAE. Nonmedian marginal quantiles do not have an hourly-mean constraint. All test months remain developmental.','',
               '| Recipe | Mean monthly MAE ratio | Pooled MAE | Mean monthly pinball ratio | Calibrated90%coverage | Calibrated90%width |',
               '|---|---:|---:|---:|---:|---:|']
        for r in summary.sort_values('mean_monthly_ratio').to_dict('records'):
            lines.append(f"| {r['recipe']} | {r['mean_monthly_ratio']:.6f} | {r['pooled_mae']:.6f} | {r['mean_monthly_pinball_ratio']:.6f} | {100*r['coverage90']:.2f}% | {r['width90']:.4f} |")
        lines += ['', 'Each fold selects/stops on the preceding month\'s first15days; remaining days calibrate intervals. No outer-test errors select a candidate, blend, calibration correction or confidence cutoff.','',
                  'Use predictions.csv.gz to inspect the validation-selected recipe quarter by quarter. It includes all eleven quantiles, calibrated90%bounds/width and calibration-relative uncertainty percentile. Narrow/middle/wide thresholds are0.25/0.75. These labels describe relative interval width, not a probability that an individual prediction is accurate. Every model\'s full raw quantiles and calibrated50/80/90/95%bounds are also in folds/<month>/selected/<recipe>/predictions.npz.','',
                  'Inspect confidence_bins.csv for whether narrow predictions actually had smaller errors, their same-bin interpolation baseline and empirical90%coverage. interval_metrics.csv includes raw/calibrated50/80/90/95%coverage/width plus whole-hour/day coverage. Pointwise calibration is not a simultaneous-day guarantee; time dependence and drift require empirical assessment.','',
                  'The validation_selector is chosen from preceding-block performance; the lowest aggregate outer-test recipe is descriptive research evidence. Quantile blends are convex averages of quantile functions, not quantiles of a mixture distribution. Their calibration is fitted separately. Earlier point-only results are preserved as historical controls. No catastrophe classifier or deployment replacement was made.','',
                  'Reproduction and fixed budget: campaigns/minimize_expanding_20261017/PROTOCOL.md and run.py. Source hashes/checkpoints/predictions/selections/calibration receipts and the independent audit are retained.']
        old=pd.read_csv(e.OLD/'summary.csv')
        prior_selector=old.loc[old.recipe=='validation_selector'].iloc[0]
        prior_blend=old.loc[old.recipe=='blend_uniform'].iloc[0]
        selector=summary.loc[summary.recipe=='validation_selector'].iloc[0]
        lines += ['', 'Training expands from October–December2025 to October2025–July2026. No training-window shortening or selection-date refit.', '',
                  f"Original pre-test selector: {prior_selector.mean_monthly_ratio:.6f}; new pre-test selector: {selector.mean_monthly_ratio:.6f}; relative MAE-ratio change: {100*(selector.mean_monthly_ratio/prior_selector.mean_monthly_ratio-1):+.3f}%.",
                  f"Original ordinary blend observed reference: {prior_blend.mean_monthly_ratio:.6f}. The new aggregate winner is descriptive and is not substituted for the pre-test selector.", '',
                  'New fits use fixed engineered physical units and clip15. Frozen standard controls are reused from the audited prior campaign; architecture, loss, learning rates and seeds match. Family winners can average the two seeds at a common learning rate. Blends are global convex quantile-function averages, selected on days1–15 and calibrated on16–end.']
        e.p.write(root/'comparison.json',dict(original_selector_ratio=float(prior_selector.mean_monthly_ratio),new_selector_ratio=float(selector.mean_monthly_ratio),
                  relative_selector_ratio_change=float(selector.mean_monthly_ratio/prior_selector.mean_monthly_ratio-1),original_blend_ratio=float(prior_blend.mean_monthly_ratio),
                  lowest_reported_recipe=summary.sort_values('mean_monthly_ratio').iloc[0]['recipe'],developmental=True,expanding_training=True))
        # Explicit alternate denominator and identical-subset zone comparisons.
        extras=[]
        for folder in folders:
            d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc
            truth=d['y'][start:];hh=d['h'][start:]
            unc=np.asarray([[np.interp(np.arange(96)/4+.125,np.arange(24)+.5,z).reshape(24,4) for z in day] for day in hh])
            corrected=np.abs(d['b'][start:]-truth);uncerr=np.abs(unc-truth)
            report=json.loads((folder/'metrics.json').read_text())
            for recipe in report['results']:
                q=np.load(folder/'selected'/recipe/'predictions.npz')['quantiles'];err=np.abs(q[...,5]-truth)
                for z in range(4):
                    extras.append(dict(month=folder.name,recipe=recipe,zone=f'ZONE_{z}',mae=float(err[:,z].mean()),
                        interpolation_mae=float(corrected[:,z].mean()),ratio=float(err[:,z].mean()/corrected[:,z].mean()),
                        uncorrected_linear_interpolation_mae=float(uncerr[:,z].mean()),
                        uncorrected_linear_interpolation_ratio=float(err[:,z].mean()/uncerr[:,z].mean())))
        pd.DataFrame(extras).to_csv(root/'monthly_zone_scores.csv',index=False)
        baselines=actual[actual.recipe=='interpolation']
        for r in summary.to_dict('records'):
            matching=actual[actual.recipe==r['recipe']]
            denominator=float(np.average(baselines.mae,weights=baselines.days))
            r['pooled_interpolation_mae']=denominator;r['pooled_mae_ratio']=r['pooled_mae']/denominator
            for objective in ['uniform','ratio']:
                if r['recipe']==f'standard_blend_{objective}':
                    close(r['mean_monthly_ratio'],old.loc[old.recipe==f'blend_{objective}','mean_monthly_ratio'].iloc[0])
            e.p.write(root/'selected_summary'/f"{r['recipe']}.json",r)
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('QUANTILE AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();audit(args.smoke)
