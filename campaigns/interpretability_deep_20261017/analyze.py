"""Posthoc explanations on an authorized server; no fitting or new predictor."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-hackaton-interpret')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/interpretability_deep_20261017'
SOURCE = ROOT / 'results/minimize_expanding_20261017'
FAMILIES = ['linear_rich', 'linear_compact', 'additive_compact', 'mlp_compact', 'cnn_rich', 'cnn2d_analog']
LABELS = ['Rich linear', 'Compact linear', 'Additive', 'MLP', 'Temporal CNN', 'Analog CNN']
AUDIT = {}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(name):
    if 'analog90' in name: return 'Similar-day residuals'
    if name.startswith('interpolation_error_'): return 'Lagged interpolation residuals'
    if name.startswith('shape_'): return 'Lagged quarter shapes'
    if 'slope' in name: return 'Hourly slopes and curvature'
    if name.startswith('today_'): return 'Full hourly curves'
    if name.startswith('local_') or name.startswith('current_zone'): return 'Local and cross-zone prices'
    if name.startswith('day_'): return 'Daily price statistics'
    if name.startswith('target_zone'): return 'Zone identity'
    return 'Calendar and hour'


def linear():
    # Copied sources have no module-scope training; never import a trainer.
    sys.path.insert(0, str(HERE / 'sources'))
    from reconstruction import read_targets, rich_features, baseline
    from linear_quantile_model import feature_names
    model_path = HERE / 'standalone_model.json'
    d = json.loads(model_path.read_text()); names = d['feature_names']
    assert names == feature_names()
    dates, y = read_targets(ROOT / 'train_prices_15min.csv')
    assert sha(ROOT / 'train_prices_15min.csv') == 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
    test = np.asarray([str(v).startswith('2026-09') for v in dates])
    h = y.mean(-1); b = baseline(h[test]); truth = y[test]
    x = rich_features(dates[test], h[test], dates, y).reshape(-1,4,24,len(names)).astype('float64')
    z = np.clip((x - np.asarray(d['mu'])) / np.asarray(d['sd']), -15, 15)
    w = 20 * np.asarray(d['weight'])[:3]; bias = 20 * np.asarray(d['bias'])[:3]
    qw = np.stack([w[0]+w[1], w[0]-w[1], -w[0]+w[2], -w[0]-w[2]], -1)
    qb = np.asarray([bias[0]+bias[1], bias[0]-bias[1], -bias[0]+bias[2], -bias[0]-bias[2]])
    contribution = z[..., :, None] * qw
    correction = contribution.sum(-2) + qb
    prediction = b + correction
    AUDIT['linear_mean_error'] = float(abs(prediction.mean(-1)-h[test]).max())
    assert AUDIT['linear_mean_error'] < 1e-8
    linear_mae = float(abs(prediction-truth).mean())
    metrics = json.loads((HERE / 'standalone_metrics.json').read_text())
    expected = [f for f in metrics['folds'] if f['plan']=='fixed_september' and f['training_months']==10][0]['test_metrics']['mae']
    AUDIT['linear_saved_mae_difference'] = abs(linear_mae-expected)
    assert AUDIT['linear_saved_mae_difference'] < 1e-8
    importance = abs(contribution).mean((0,1,2,4))
    rows = [dict(feature=n, group=group(n), mean_abs_contribution=float(importance[j])) for j,n in enumerate(names)]
    rows.sort(key=lambda r:r['mean_abs_contribution'],reverse=True); table('linear_feature_contributions.csv',rows)
    groups = sorted(set(group(n) for n in names)); gc = {}
    gr = []
    for g in groups:
        gc[g] = contribution[..., [group(n)==g for n in names], :].sum(-2)
        for zone in [-1,0,1,2,3]:
            c = gc[g] if zone<0 else gc[g][:,zone]
            gr.append(dict(group=g, zone='ALL' if zone<0 else f'ZONE_{zone}',mean_abs_net_contribution=float(abs(c).mean())))
    table('linear_group_contributions.csv',gr)
    gain = abs(b-truth).mean(-1)-abs(prediction-truth).mean(-1)
    choices = [('Largest modeled correction', np.unravel_index(abs(correction).mean(-1).argmax(), gain.shape)),
               ('Largest improvement (hindsight)', np.unravel_index(gain.argmax(),gain.shape)),
               ('Largest harm (hindsight)',np.unravel_index(gain.argmin(),gain.shape))]
    examples=[]; erows=[]
    fig,axes=plt.subplots(3,2,figsize=(14,10),gridspec_kw={'width_ratios':[1,1.2]})
    for i,(label,idx) in enumerate(choices):
        di,zi,hi=idx
        c = contribution[idx]; order=np.argsort(abs(c).mean(-1))[::-1][:8]
        contrast = z[idx] @ w.T + bias
        ex=dict(label=label,date=str(dates[test][di]),zone=f'ZONE_{zi}',hour=int(hi+1),
                baseline=b[idx].tolist(),predicted=prediction[idx].tolist(),observed=truth[idx].tolist(),
                a_u_v=contrast.tolist(),bias_quarters=qb.tolist(),
                top_features=[dict(feature=names[j],quarter_contribution=c[j].tolist()) for j in order],
                grouped_contributions={g:gc[g][idx].tolist() for g in groups})
        examples.append(ex)
        for k in range(4):
            erows.append(dict(date=ex['date'],zone=ex['zone'],hour=hi+1,quarter=k+1,baseline=b[idx][k],
                              predicted=prediction[idx][k],observed=truth[idx][k],correction=correction[idx][k]))
        ax=axes[i,0]; ax.plot(range(1,5),truth[idx],'o-',label='Observed',color='#d85c30')
        ax.plot(range(1,5),b[idx],'s--',label='Interpolation',color='#888888')
        ax.plot(range(1,5),prediction[idx],'o-',label='Linear',color='#2366a8')
        ax.set_title(f"{label}\n{ex['date']} {ex['zone']} hour {hi+1}",fontsize=10); ax.set_xticks(range(1,5));ax.set_ylabel('EUR/MWh');ax.grid(alpha=.2)
        matrix=np.asarray([gc[g][idx] for g in groups]+[qb]);ax=axes[i,1]
        vmax=max(1,float(abs(matrix).max()));im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        ax.set_yticks(range(len(groups)+1),groups+['Bias'],fontsize=8);ax.set_xticks(range(4),['Q1','Q2','Q3','Q4'])
        for a in range(len(matrix)):
            for k in range(4):ax.text(k,a,f'{matrix[a,k]:.1f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix[a,k])>.65*vmax else 'black')
        ax.set_title('Exact contributions to interpolation correction',fontsize=10);fig.colorbar(im,ax=ax,label='EUR/MWh',fraction=.03,pad=.02)
    axes[0,0].legend(fontsize=8);fig.suptitle('Standalone linear model: exact September explanations\nExtreme illustrative cases; improvement/harm selected with hindsight',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'linear_examples.png',dpi=160);fig.savefig(OUT/'linear_examples.svg');plt.close(fig)
    table('linear_examples.csv',erows);write('linear_examples.json',examples)
    write('linear_summary.json',dict(model_sha256=sha(model_path),mae=linear_mae,baseline_mae=float(abs(b-truth).mean()),
          top_features=rows[:12],groups=sorted([r for r in gr if r['zone']=='ALL'],key=lambda r:r['mean_abs_net_contribution'],reverse=True)))
    print('LINEAR COMPLETE',linear_mae,flush=True)


def ensemble():
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    records=[];complement=[];uncertainty=[];removals=[];examples=[];signed={f:[] for f in FAMILIES}
    raw=np.load(SOURCE/'raw.npz');raw_dates=raw['dates'];raw_h=raw['h']
    for folder in sorted((SOURCE/'folds').iterdir()):
        if not folder.is_dir():continue
        month=folder.name; selection=json.loads((folder/'selection.json').read_text())['by_recipe']
        with np.load(folder/'selected/fixed_blend_ratio/predictions.npz') as a:
            y=a['observed'];dates=a['dates'];blend=a['quantiles'][...,5];pct=a['uncertainty_percentile']
            lo=a['lower_90'];hi=a['upper_90']
        ids=np.searchsorted(raw_dates,dates);np.testing.assert_array_equal(raw_dates[ids],dates)
        # data.npz baseline is the campaign's audited interpolation control.
        with np.load(folder/'data.npz') as data: b=data['b'][ids]; h=data['h'][ids];nt=int(data['nt'])
        np.testing.assert_allclose(y,raw['y'][ids],atol=0,rtol=0)
        weights=selection['fixed_blend_ratio']['weights']; preds={}
        for f in FAMILIES:
            name='fixed_'+f+'_ratio'
            with np.load(folder/'selected'/name/'predictions.npz') as a:
                np.testing.assert_array_equal(a['dates'],dates);np.testing.assert_array_equal(a['observed'],y)
                preds[f]=a['quantiles'][...,5]
            signed[f].append((preds[f]-y).ravel())
        e=np.asarray([preds[f]-y for f in FAMILIES]);wv=np.asarray([weights.get('fixed_'+f+'_ratio',0.) for f in FAMILIES])
        replay=np.einsum('i,idzhq->dzhq',wv,np.asarray([preds[f] for f in FAMILIES]))
        delta=float(abs(replay-blend).max());assert delta<1e-8
        AUDIT['champion_replay_max_error']=max(AUDIT.get('champion_replay_max_error',0),delta)
        assert abs(blend.mean(-1)-h).max()<1e-8
        weighted=np.einsum('i,idzhq->dzhq',wv,abs(e));be=abs(blend-y);base=abs(b-y)
        cancel=weighted-be;assert cancel.min()>-1e-8
        active=wv>0
        mixed=(e[active].min(0)<0)&(e[active].max(0)>0)
        denom=float(base.mean())
        complement.append(dict(month=month,blend_ratio=float(be.mean()/denom),weighted_member_ratio=float(weighted.mean()/denom),
            cancellation_gain_ratio=float(cancel.mean()/denom),cancellation_eur=float(cancel.mean()),
            opposite_sign_quarters_percent=float(100*mixed.mean()),
            blend_beats_every_active_expert_hours_percent=float(100*(be.mean(-1)<abs(e[active]).mean(-1).min(0)-1e-10).mean())))
        for i,f in enumerate(FAMILIES):
            if wv[i]<=0 or wv[i]>=1:continue
            without=(replay-wv[i]*preds[f])/(1-wv[i]);mae=float(abs(without-y).mean())
            removals.append(dict(month=month,family=f,weight=wv[i],blend_mae=float(be.mean()),without_member_mae=mae,delta_mae=mae-float(be.mean())))
        trainh=raw_h[:nt];tp=np.pad(trainh,((0,0),(0,0),(1,1)),mode='edge')
        cut=float(np.quantile(np.maximum(abs(trainh-tp[:,:,:24]),abs(tp[:,:,2:]-trainh)),.25))
        hp=np.pad(h,((0,0),(0,0),(1,1)),mode='edge');left=h-hp[:,:,:24];right=hp[:,:,2:]-h
        ramp=np.maximum(abs(left),abs(right));shape=np.where(ramp<=cut,'Flat',np.where((left>0)&(right<0),'Peak',np.where((left<0)&(right>0),'Trough',np.where(left+right>=0,'Rising','Falling'))))
        masks=[('All','All',np.ones(y.shape[:-1],bool))]
        masks += [('Zone',f'ZONE_{z}',np.broadcast_to(np.arange(4)[None,:,None]==z,y.shape[:-1])) for z in range(4)]
        masks += [('Curve',s,shape==s) for s in ['Flat','Rising','Falling','Peak','Trough']]
        masks += [('Hour',str(t+1),np.broadcast_to(np.arange(24)[None,None,:]==t,y.shape[:-1])) for t in range(24)]
        masks += [('ZoneCurve',f'ZONE_{z}/{s}',(shape==s)&np.broadcast_to(np.arange(4)[None,:,None]==z,y.shape[:-1])) for z in range(4) for s in ['Flat','Rising','Falling','Peak','Trough']]
        for g,val,mask in masks:
            if not mask.any():continue
            den=float(base[mask].mean());assert den>0
            for f,pred in list(preds.items())+[('blend',blend)]:
                mae=float(abs(pred[mask]-y[mask]).mean())
                records.append(dict(month=month,group=g,value=val,family=f,hours=int(mask.sum()),mae=mae,
                                    baseline_mae=den,ratio=mae/den))
        for label,mask in [('narrow',pct<=.25),('middle',(pct>.25)&(pct<.75)),('wide',pct>=.75)]:
            den=float(base[mask].mean());mae=float(be[mask].mean())
            uncertainty.append(dict(month=month,label=label,n=int(mask.sum()),mae=mae,baseline_mae=den,ratio=mae/den,
                                    improvement_percent=100*(1-mae/den),coverage90=float(((y>=lo)&(y<=hi))[mask].mean()),width90=float((hi-lo)[mask].mean())))
        # Explicit hindsight examples: strongest cancellation and averaging harm per month.
        for label,score in [('Largest cancellation',cancel.mean(-1)),('Largest harm versus best active expert',be.mean(-1)-abs(e[active]).mean(-1).min(0))]:
            idx=np.unravel_index(score.argmax(),score.shape);di,zi,ti=idx
            examples.append(dict(month=month,label=label,date=str(dates[di]),zone=f'ZONE_{zi}',hour=int(ti+1),
                observed=y[idx].tolist(),baseline=b[idx].tolist(),blend=blend[idx].tolist(),
                members={f:dict(weight=float(wv[i]),predicted=preds[f][idx].tolist(),error=e[i][idx].tolist()) for i,f in enumerate(FAMILIES) if active[i]},
                weighted_member_mae=float(weighted[idx].mean()),blend_mae=float(be[idx].mean()),cancellation=float(cancel[idx].mean())))
        print('CHAMPION',month,complement[-1],flush=True)
    assert len(complement)==8
    table('conditional_monthly.csv',records);table('complementarity.csv',complement);table('member_removal.csv',removals);table('uncertainty_monthly.csv',uncertainty)
    agg=[]
    keys=sorted(set((r['group'],r['value'],r['family']) for r in records))
    for key in keys:
        rows=[r for r in records if (r['group'],r['value'],r['family'])==key];n=sum(r['hours'] for r in rows)
        agg.append(dict(group=key[0],value=key[1],family=key[2],months=len(rows),hours=n,
                        equal_month_ratio=float(np.mean([r['ratio'] for r in rows])),
                        pooled_mae=sum(r['mae']*r['hours'] for r in rows)/n,
                        pooled_baseline_mae=sum(r['baseline_mae']*r['hours'] for r in rows)/n))
    table('conditional_summary.csv',agg)
    us=[]
    for label in ['narrow','middle','wide']:
        rows=[r for r in uncertainty if r['label']==label];n=sum(r['n'] for r in rows)
        us.append(dict(label=label,equal_month_ratio=float(np.mean([r['ratio'] for r in rows])),
            equal_month_mae=float(np.mean([r['mae'] for r in rows])),equal_month_baseline_mae=float(np.mean([r['baseline_mae'] for r in rows])),
            pooled_mae=sum(r['n']*r['mae'] for r in rows)/n,pooled_baseline_mae=sum(r['n']*r['baseline_mae'] for r in rows)/n,
            months_with_ratio_below_one=sum(r['ratio']<1 for r in rows)))
    correlations=np.corrcoef([np.concatenate(signed[f]) for f in FAMILIES]);write('signed_error_correlations.json',dict(families=FAMILIES,matrix=correlations.tolist()))
    summary=dict(mean_blend_ratio=float(np.mean([r['blend_ratio'] for r in complement])),
        mean_weighted_member_ratio=float(np.mean([r['weighted_member_ratio'] for r in complement])),
        mean_cancellation_gain_ratio=float(np.mean([r['cancellation_gain_ratio'] for r in complement])),
        pooled_cancellation_eur=float(np.average([r['cancellation_eur'] for r in complement],weights=[28,31,29,31,30,31,31,30])),
        mean_opposite_sign_percent=float(np.mean([r['opposite_sign_quarters_percent'] for r in complement])),
        uncertainty=us,curve=[r for r in agg if r['group']=='Curve' and r['family']=='blend'],
        zones=[r for r in agg if r['group']=='Zone' and r['family']=='blend'])
    assert abs(summary['mean_blend_ratio']-.7079682867567635)<1e-9
    write('ensemble_summary.json',summary);write('ensemble_examples.json',examples)
    chosen=[max((x for x in examples if x['label']==lab),key=lambda x:x['cancellation'] if lab=='Largest cancellation' else x['blend_mae']-min(np.mean(abs(np.asarray(m['error']))) for m in x['members'].values())) for lab in ['Largest cancellation','Largest harm versus best active expert']]
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,ex in zip(axes,chosen):
        for f,m in ex['members'].items():ax.plot(range(1,5),m['predicted'],'o--',alpha=.65,label=f'{LABELS[FAMILIES.index(f)]} ({m["weight"]:.2f})')
        ax.plot(range(1,5),ex['blend'],'s-',color='#2366a8',lw=2,label='Blend');ax.plot(range(1,5),ex['observed'],'o-',color='#d85c30',lw=2,label='Observed')
        ax.set_title(f"{ex['label']}\n{ex['date']} {ex['zone']} hour {ex['hour']}",fontsize=10);ax.set_xticks(range(1,5));ax.set_ylabel('EUR/MWh');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Champion: averaging can cancel errors and dilute a strong expert\nHindsight extremes across eight developmental months',fontsize=12);fig.tight_layout();fig.savefig(OUT/'ensemble_examples.png',dpi=160);fig.savefig(OUT/'ensemble_examples.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    zr=summary['zones'];axes[0].bar([r['value'] for r in zr],[100*(1-r['equal_month_ratio']) for r in zr],color='#2366a8');axes[0].set_ylabel('MAE reduction vs interpolation (%)');axes[0].set_title('Equal-month zone gains')
    cr=summary['curve'];axes[1].bar([r['value'] for r in cr],[100*(1-r['equal_month_ratio']) for r in cr],color='#438e73');axes[1].set_title('Known hourly curve context');axes[1].tick_params(axis='x',rotation=30)
    x=np.arange(3);axes[2].bar(x-.18,[r['equal_month_mae'] for r in us],.36,label='Champion',color='#2366a8');axes[2].bar(x+.18,[r['equal_month_baseline_mae'] for r in us],.36,label='Interpolation',color='#999999');axes[2].set_xticks(x,[r['label'] for r in us]);axes[2].set_ylabel('Equal-month mean MAE (EUR/MWh)');axes[2].set_title('Uncertainty groups');axes[2].legend(fontsize=8)
    fig.suptitle('Current champion: matched saved forecasts, February–September',fontsize=13);fig.tight_layout();fig.savefig(OUT/'zone_curve_uncertainty.png',dpi=160);fig.savefig(OUT/'zone_curve_uncertainty.svg');plt.close(fig)
    print('ENSEMBLE SUMMARY',json.dumps(summary),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    linear();ensemble()
    AUDIT.update(passed=True,folds=8,training=False,scope='Posthoc existing checkpoint/forecast explanations; no new model selection',
                 analysis_sha256=sha(Path(__file__)),source_audit_sha256=sha(SOURCE/'audit.json'))
    write('audit.json',AUDIT)
    print('INTERPRETABILITY COMPLETE',json.dumps(AUDIT),flush=True)


if __name__=='__main__':main()
