"""Posthoc model-specialization diagnostics; no new fitting or routing policy."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'results/monthly_quantiles_20261017'
OUT=SOURCE/'ensemble_analysis'
FAMILIES=['linear_rich','linear_compact','additive_compact','mlp_compact','cnn_rich','cnn2d_analog']
LABELS=['Rich linear','Compact linear','Additive','MLP','Temporal CNN','Analog 2D CNN']


def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    data=np.load(SOURCE/'raw.npz');months=np.asarray([str(d)[:7] for d in data['dates']])
    rows=[];weight_rows=[];diagnostics=[];contexts=[];signed={};thresholds=[]
    maximum_reconstruction_error=0.
    for folder in sorted((SOURCE/'folds').iterdir()):
        if not folder.is_dir():continue
        selection=json.loads((folder/'selection.json').read_text())['by_recipe']
        report=json.loads((folder/'metrics.json').read_text());meta=report['manifest'];month=meta['test']
        d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc
        y=d['y'][start:];h=d['h'][start:];b=d['b'][start:];baseline_error=np.abs(b-y).mean(-1)
        train=(months>=meta['train_start'])&(months<=meta['train_end'])
        def slopes(values):
            padded=np.pad(values,((0,0),(0,0),(1,1)),mode='edge')
            return values-padded[:,:,:24],padded[:,:,2:]-values
        tl,tr=slopes(data['h'][train]);train_ramp=np.maximum(np.abs(tl),np.abs(tr))
        cut=float(np.quantile(train_ramp,.25));left,right=slopes(h);ramp=np.maximum(np.abs(left),np.abs(right))
        shape=np.where(ramp<=cut,'Flat',np.where((left>0)&(right<0),'Peak',np.where((left<0)&(right>0),'Trough',np.where(left+right>=0,'Rising','Falling'))))
        thresholds.append(dict(month=month,flat_max_ramp=cut))
        base_masks=[('All','All',np.ones(y.shape[:-1],bool))]
        base_masks += [('Zone',f'ZONE_{z}',np.broadcast_to(np.arange(4)[None,:,None]==z,y.shape[:-1])) for z in range(4)]
        base_masks += [('Hour',str(hour+1),np.broadcast_to(np.arange(24)[None,None,:]==hour,y.shape[:-1])) for hour in range(24)]
        base_masks += [('Curve',name,shape==name) for name in ['Flat','Rising','Falling','Peak','Trough']]
        for mode in ['uniform','ratio']:
            values=[];errors=[];by_family={}
            for family in FAMILIES:
                name=family+'_'+mode;p=np.load(folder/'selected'/name/'predictions.npz')
                np.testing.assert_array_equal(p['observed'],y)
                q=p['quantiles'];pred=q[...,5];err=np.abs(pred-y)
                np.testing.assert_allclose(err.mean(),report['results'][name]['mae'],rtol=0,atol=1e-9)
                assert np.abs(pred.mean(-1)-h).max()<1e-8
                values.append(pred);errors.append(err.mean(-1));by_family[name]=q
                signed.setdefault((mode,family),[]).append((pred-y).ravel())
            predictions=np.asarray(values);hour_errors=np.asarray(errors)
            blend_name='blend_'+mode;weights=selection[blend_name]['weights']
            ensemble=np.load(folder/'selected'/blend_name/'predictions.npz')
            reconstructed=sum(by_family[name]*weight for name,weight in weights.items())
            delta=float(np.abs(reconstructed-ensemble['quantiles']).max());assert delta<1e-9
            maximum_reconstruction_error=max(maximum_reconstruction_error,delta)
            blend_errors=np.abs(ensemble['quantiles'][...,5]-y).mean(-1)
            winner=hour_errors.argmin(0);oracle=hour_errors.min(0)
            weighted_member_error=sum(hour_errors[FAMILIES.index(name.removesuffix('_'+mode))]*w for name,w in weights.items())
            wholemonthbaseline=float(baseline_error.mean())
            diagnostics.append(dict(month=month,mode=mode,blend_ratio=float(blend_errors.mean()/wholemonthbaseline),
                                    oracle_whole_hour_ratio=float(oracle.mean()/wholemonthbaseline),
                                    weighted_member_ratio=float(weighted_member_error.mean()/wholemonthbaseline),
                                    cancellation_gain=float((weighted_member_error.mean()-blend_errors.mean())/wholemonthbaseline),
                                    blend_beats_every_expert_hours_percent=float(100*(blend_errors<hour_errors.min(0)-1e-10).mean())))
            for family,label in zip(FAMILIES,LABELS):
                weight_rows.append(dict(month=month,mode=mode,family=family,label=label,weight=weights.get(family+'_'+mode,0.)))
            confidence=ensemble['confidence_bin']
            # Hour confidence summarizes all four prediction widths; thresholds are
            # calibration percentiles already frozen by the original campaign.
            uncertainty=ensemble['uncertainty_percentile'].mean(-1)
            masks=base_masks+[('Uncertainty',name,mask) for name,mask in [
                ('Narrow',uncertainty<=.25),('Middle',(uncertainty>.25)&(uncertainty<.75)),('Wide',uncertainty>=.75)]]
            for group,value,mask in masks:
                count=int(mask.sum())
                if count<20:continue
                denom=float(baseline_error[mask].mean())
                if denom<=1e-12:continue
                records=[]
                for index,family in enumerate(FAMILIES):
                    mae=float(hour_errors[index][mask].mean());wins=float((winner[mask]==index).mean())
                    row=dict(month=month,mode=mode,group=group,value=value,family=family,hours=count,
                             mae=mae,interpolation_mae=denom,ratio=mae/denom,hindsight_hour_win_share=wins)
                    rows.append(row);records.append(dict(id=family,r=round(mae/denom,7),mae=round(mae,6),wins=round(wins,6)))
                mae=float(blend_errors[mask].mean())
                rows.append(dict(month=month,mode=mode,group=group,value=value,family='blend',hours=count,mae=mae,
                                 interpolation_mae=denom,ratio=mae/denom,hindsight_hour_win_share=None))
                records.append(dict(id='blend',r=round(mae/denom,7),mae=round(mae,6),wins=None))
                contexts.append(dict(month=month,mode=mode,group=group,value=value,hours=count,
                                     baseline=round(denom,6),oracle=round(float(oracle[mask].mean()/denom),7),models=records))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'conditional_monthly_scores.csv',index=False)
    aggregate=[]
    for key,g in frame.groupby(['mode','group','value','family']):
        mode,group,value,family=key
        aggregate.append(dict(mode=mode,group=group,value=value,family=family,months=len(g),hours=int(g.hours.sum()),
                              mean_monthly_ratio=float(g.ratio.mean()),pooled_mae=float(np.average(g.mae,weights=g.hours)),
                              hindsight_hour_win_share=float(np.average(g.hindsight_hour_win_share,weights=g.hours)) if family!='blend' else None))
    aggregated=pd.DataFrame(aggregate);aggregated.to_csv(OUT/'conditional_summary.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(OUT/'weights.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'complementarity.csv',index=False)
    correlations={}
    for mode in ['uniform','ratio']:
        correlations[mode]=np.corrcoef([np.concatenate(signed[mode,f]) for f in FAMILIES]).tolist()
    payload=dict(families=[dict(id=f,label=l) for f,l in zip(FAMILIES,LABELS)],months=sorted(set(v['month'] for v in contexts)),
                 weights=[dict(month=month,mode=mode,values=[next(w['weight'] for w in weight_rows if w['month']==month and w['mode']==mode and w['family']==f) for f in FAMILIES]) for month,mode in itertools.product(sorted(set(v['month'] for v in contexts)),['uniform','ratio'])],
                 contexts=contexts,diagnostics=diagnostics,flat_thresholds=thresholds,
                 definitions=dict(ratio='Mean across monthly MAE / interpolation MAE on the identical context subset',
                                  weights='Fixed per test fold, selected on preceding-month days1-15; used for all inputs/quantile levels',
                                  winner='Hindsight winner among six experts for each whole zone/hour; not a deployable router',
                                  shape='Flat if max absolute neighboring slope is below fitting-history 25th percentile; otherwise peak/trough/rising/falling'))
    (OUT/'explorer.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
    (OUT/'signed_error_correlations.json').write_text(json.dumps(dict(families=FAMILIES,matrices=correlations),indent=2)+'\n')
    summary={mode:dict(mean_blend_ratio=float(np.mean([v['blend_ratio'] for v in diagnostics if v['mode']==mode])),
                       hindsight_oracle_whole_hour_ratio=float(np.mean([v['oracle_whole_hour_ratio'] for v in diagnostics if v['mode']==mode])),
                       mean_weighted_member_ratio=float(np.mean([v['weighted_member_ratio'] for v in diagnostics if v['mode']==mode])),
                       mean_cancellation_gain=float(np.mean([v['cancellation_gain'] for v in diagnostics if v['mode']==mode])),
                       blend_beats_every_expert_hours_percent=float(np.mean([v['blend_beats_every_expert_hours_percent'] for v in diagnostics if v['mode']==mode]))) for mode in ['uniform','ratio']}
    # Every context mask is a known-input condition; the winning expert still uses
    # observed targets, so these diagnostics must never silently become a policy.
    audit=dict(passed=True,folds=8,modes=2,experts=6,maximum_quantile_blend_reconstruction_error=maximum_reconstruction_error,
               sources=dict(source_audit=hashlib.sha256((SOURCE/'audit.json').read_bytes()).hexdigest(),
                            analysis_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
               checks='Official prediction identities, constituent MAE parity, q50 mean conservation, all-quantile weighted blend identity; curve thresholds use fitting hourly inputs only; no parameter or routing-policy fit')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Ensemble weights and model specialization','',
           'Posthoc analysis of completed quantile predictions; no router was fitted and no forecast was changed. Current weights are fixed for a whole test month and selected using the preceding first15validationdays. Quantile functions are averaged at every level, then blend intervals are calibrated separately.','',
           '## Composition of the best aggregate recipe (ordinary training)','',
           '| Test month | Components |','|---|---|']
    for month in payload['months']:
        ws=[w for w in weight_rows if w['month']==month and w['mode']=='uniform' and w['weight']>0]
        lines.append('| '+month+' | '+'; '.join(f"{v['label']} {100*v['weight']:.1f}%" for v in ws)+' |')
    lines+=['','## Conditional scores','',
            'Within each context, the ratio divides model MAE by interpolation MAE on that same subset and averages monthly ratios. Counts are whole zone/hours, each containing four quarters. These conditional scores are distinct from the overall eight-month metric. Curves use fitting-only slope thresholds. All these comparisons remain developmental and posthoc.','',
            '| Context | Best single expert (ordinary training) | Ratio | Blend ratio | Months won by that expert |',
            '|---|---|---:|---:|---:|']
    for group in ['Zone','Curve','Uncertainty']:
        for value in sorted(aggregated[(aggregated['mode']=='uniform')&(aggregated.group==group)].value.unique()):
            subset=aggregated[(aggregated['mode']=='uniform')&(aggregated.group==group)&(aggregated.value==value)]
            best=subset[subset.family!='blend'].sort_values('mean_monthly_ratio').iloc[0]
            blend=subset[subset.family=='blend'].iloc[0]
            wins=0;months_count=0
            for context in [c for c in contexts if c['mode']=='uniform' and c['group']==group and c['value']==value]:
                winner=min((r for r in context['models'] if r['id']!='blend'),key=lambda r:r['r'])
                wins+=winner['id']==best.family;months_count+=1
            lines.append(f"| {group}: {value} | {LABELS[FAMILIES.index(best.family)]} | {best.mean_monthly_ratio:.6f} | {blend.mean_monthly_ratio:.6f} | {wins}/{months_count} |")
    lines+=['','## Complementarity and routing bounds','',json.dumps(summary,indent=2),'',
            'The weighted-member ratio is the error if averaging constituent absolute errors rather than predictions. Its reduction after prediction averaging quantifies cancellation, not a router. The hindsight oracle chooses the expert with lowest true four-quarter error for each whole zone/hour, preserving the hourly mean. It reads the answer and is only an optimistic diagnostic bound. It is not a measured attainable routing gain. Some hours are better served by the blend than every expert.','',
            'An actual interpretable gate could use a small tree or additive gate from known current hourly slopes/price/spread, prior interpolation residuals and expert uncertainty/disagreement. It must be trained using earlier out-of-sample expert errors, selected chronologically, and tested on the next month. Route or mix all four quarters of each hour consistently and calibrate the final gated quantiles separately. No such gate was trained in this analysis.','',
            'Files: weights.csv, conditional_monthly_scores.csv, conditional_summary.csv, complementarity.csv, signed_error_correlations.json, explorer.json and audit.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,context_report=lines[lines.index('## Conditional scores'):lines.index('## Complementarity and routing bounds')],audit=audit),indent=2))


if __name__=='__main__':main()
