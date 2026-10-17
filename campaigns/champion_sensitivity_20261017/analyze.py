"""Frozen champion sensitivity, with underlying inputs rebuilt coherently."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-champion-sensitivity')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'campaigns/minimize_expanding_20261017'))
import train as e
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT/'results/minimize_expanding_20261017'
OUT = ROOT/'results/champion_sensitivity_20261017'
OUT.mkdir(exist_ok=True, parents=True)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value): (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def inputs(dates, hourly, hd, hy):
    raw=e.p.features(dates.astype(object), hourly, hd.astype(object), hy)
    data={k:np.clip(raw[k],-15,15).astype('float32') for k in ['rich','compact','cnn','curve']}
    data['compact']=np.concatenate([data['compact'],np.broadcast_to(np.eye(4)[None,:,None,:],(len(dates),4,24,4))],-1).astype('float32')
    return data

def main():
    e.p.setup()
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for name,h in manifest['code_hashes'].items(): assert sha(ROOT/name)==h, name
    assert sha(ROOT/'train_prices_15min.csv')==manifest['source_sha256']
    assert sha(SOURCE/'raw.npz')==manifest['feature_source_sha256']
    raw=np.load(SOURCE/'raw.npz'); hd=raw['dates']; hy=raw['y']; hh=raw['h']
    hb=e.p.baseline(hh)
    interventions=[('history_residual_half', 'history',.5),('history_residual_zero','history',0.),
                   ('current_curve_90pct','curve',.9),('current_curve_110pct','curve',1.1)]
    write('protocol.json',dict(sample_days=[3,10,17,24],sampling='whole dates/all zones and hours; fixed before outcomes',
          interventions=interventions,history_reference='same historical hourly curves, mean-corrected interpolation',
          curve_reference='same-day per-zone mean; 0.9/1.1 deviations, historical observations unchanged',
          source_manifest_sha256=sha(SOURCE/'manifest.json'),training=False,
          caveats='Fixed-checkpoint sensitivity, not retrained ablation or causality. Intervention magnitudes are not comparable feature importance. Only history interventions retain original-target MAE meaning, with distribution-shift caveat.'))
    rows=[]; audits=[]; checkpoints={}; examples=[]
    for folder in sorted((SOURCE/'folds').iterdir()):
        if not folder.is_dir(): continue
        month=folder.name; d=np.load(folder/'data.npz')
        sel=json.loads((folder/'selection.json').read_text())['by_recipe']
        saved=np.load(folder/'selected/fixed_blend_ratio/predictions.npz')
        dates=saved['dates']; mask=np.array([int(v[8:10]) in [3,10,17,24] for v in dates])
        dates=dates[mask]; ids=np.searchsorted(hd,dates); np.testing.assert_array_equal(hd[ids],dates)
        hourly=hh[ids]; anchor=raw['b'][ids]; truth=hy[ids]
        original=inputs(dates,hourly,hd,hy)
        did=np.searchsorted(d['dates'],dates)
        for k in original: np.testing.assert_allclose(original[k],d[k][did],rtol=0,atol=1e-7)
        # Future/current quarters must not affect any engineered feature.
        cutoff=dates[0]; changed=hy.copy(); changed[hd>=cutoff]+=np.arange(4)*123456
        probe=inputs(dates[:1],hourly[:1],hd,changed)
        truncated=inputs(dates[:1],hourly[:1],hd[hd<cutoff],hy[hd<cutoff])
        for k in original:
            np.testing.assert_allclose(probe[k],original[k][:1],atol=1e-7,rtol=0)
            np.testing.assert_allclose(truncated[k],original[k][:1],atol=1e-7,rtol=0)
        models=[]
        for family,weight in sel['fixed_blend_ratio']['weights'].items():
            for cid,seedweight in sel[family]['components'].items():
                cp=folder/'trials'/cid/'model.pt'; checkpoints[str(cp.relative_to(ROOT))]=sha(cp)
                checkpoint=torch.load(cp,weights_only=False,map_location='cpu')
                net=e.Quantiles(checkpoint['config'],d).cuda().eval(); net.load_state_dict(checkpoint['state'])
                models.append((net,weight*seedweight))
        def predict(data,base):
            # Retain the original inference batch shape: GPU kernels otherwise
            # introduce ~1e-5 EUR rounding differences for small sampled batches.
            nt=int(d['nt']); local=did-nt
            batch={k:d[k][nt:].copy() for k in ['rich','compact','cnn','curve']}
            for k in batch: batch[k][local]=data[k]
            batch_base=d['b'][nt:].copy();batch_base[local]=base
            result=np.zeros(batch_base.shape+(11,))
            with torch.no_grad():
                for net,w in models:
                    x=torch.as_tensor(batch[net.feature],device='cuda'); curve=torch.as_tensor(batch['curve'],device='cuda')
                    result+=w*e.physical(net(x,curve).cpu().numpy(),batch_base)
            result=result[local]
            assert np.isfinite(result).all() and np.diff(result,axis=-1).min()>=-1e-9
            assert abs(result[...,5].mean(-1)-base.mean(-1)).max()<1e-4
            return result
        q=predict(original,anchor); delta=float(abs(q-saved['quantiles'][mask]).max())
        print('REPLAY',month,delta,flush=True)
        assert delta<1e-6, (month,delta)
        original_median=q[...,5]; original_mae=float(abs(original_median-truth).mean())
        audits.append(dict(month=month,dates=dates.tolist(),replay_max_error=delta,causal_probe=True,
                           hourly_mean_error=float(abs(original_median.mean(-1)-hourly).max())))
        for name,kind,scale in interventions:
            if kind=='history':
                intervention_y=hb+scale*(hy-hb)
                np.testing.assert_allclose(intervention_y.mean(-1),hh,atol=1e-9,rtol=0)
                data=inputs(dates,hourly,hd,intervention_y); newanchor=anchor
            else:
                center=hourly.mean(-1,keepdims=True); newhourly=center+scale*(hourly-center)
                data=inputs(dates,newhourly,hd,hy); newanchor=e.p.baseline(newhourly)
            newq=predict(data,newanchor); newmedian=newq[...,5]
            forecast_delta=newmedian-original_median; baseline_delta=newanchor-anchor
            correction_delta=forecast_delta-baseline_delta
            row=dict(month=month,intervention=name,kind=kind,days=len(dates),
                     forecast_change=float(abs(forecast_delta).mean()),baseline_change=float(abs(baseline_delta).mean()),
                     correction_change=float(abs(correction_delta).mean()),original_mae=original_mae,
                     original_target_mae=float(abs(newmedian-truth).mean()) if kind=='history' else None,
                     median_width_change=float(abs((newq[...,9]-newq[...,1])-(q[...,9]-q[...,1])).mean()))
            rows.append(row)
            # Fixed first sampled day and fixed zone/hour, independent of errors.
            examples.append(dict(month=month,intervention=name,date=str(dates[0]),zone='ZONE_0',hour=18,
                                 original=original_median[0,0,17].tolist(),intervened=newmedian[0,0,17].tolist(),
                                 anchor=anchor[0,0,17].tolist(),newanchor=newanchor[0,0,17].tolist()))
            print('COMPLETE',month,name,json.dumps(row),flush=True)
        del models
        torch.cuda.empty_cache()
        write('progress.json',dict(completed_months=len(audits),total_months=8))
    with (OUT/'monthly.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for name,kind,scale in interventions:
        rr=[r for r in rows if r['intervention']==name]
        summary.append(dict(intervention=name,kind=kind,months=len(rr),
               forecast_change=float(np.mean([r['forecast_change'] for r in rr])),
               baseline_change=float(np.mean([r['baseline_change'] for r in rr])),
               correction_change=float(np.mean([r['correction_change'] for r in rr])),
               mae_change=float(np.mean([r['original_target_mae']-r['original_mae'] for r in rr])) if kind=='history' else None))
    write('summary.json',summary);write('examples.json',examples)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=['History residuals ×0.5','History residuals ×0','Current curve ×0.9','Current curve ×1.1']
    axes[0].barh(labels,[r['forecast_change'] for r in summary],color='#3977c4')
    axes[0].set_xlabel('Mean absolute forecast change (EUR/MWh)');axes[0].invert_yaxis()
    axes[0].set_title('Response to coherent input interventions')
    axes[1].barh(labels,[r['correction_change'] for r in summary],color='#16816a')
    axes[1].set_xlabel('Mean absolute learned-correction change (EUR/MWh)');axes[1].invert_yaxis()
    axes[1].set_title('After subtracting interpolation-anchor changes')
    fig.suptitle('Frozen research champion: four fixed dates per month, eight monthly folds')
    fig.text(.04,.02,'Sensitivity magnitudes depend on intervention size; not causal effects or a feature-importance ranking.',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(OUT/'sensitivity.png',dpi=170);plt.close(fig)
    lines=['# Frozen champion input sensitivity','',
           'Completed on authorized research GPU. Four prespecified dates (3, 10, 17, 24) per test month, all zones/hours. No fitting, recipe selection or submission changes.',
           '', 'Original checkpoint blends replayed all eleven saved quantiles within 1e-6 EUR/MWh. Source hashes and causal feature probes passed. Every intervention retained ordered finite quantiles and conserved q50 hourly means.',
           '', '| Intervention | Forecast change | Learned-correction change | Original-target MAE change |',
           '|---|---:|---:|---:|']
    for r in summary:
        mae='Not a counterfactual target' if r['mae_change'] is None else f"{r['mae_change']:+.4f}"
        lines.append(f"| {r['intervention']} | {r['forecast_change']:.4f} | {r['correction_change']:.4f} | {mae} |")
    lines+=['','Values are equal-month means in EUR/MWh. History interventions shrink observed historical deviations from mean-corrected interpolation while preserving all historical hourly means. Current-curve interventions scale deviations from each zone’s daily mean with history unchanged; the anchor and all dependent features/analog retrieval are rebuilt.',
            '', 'Historical masking can be out of distribution. Curves are synthetic coherent sensitivity probes, not validated physical counterfactuals. Intervention sizes differ: do not rank input importance from these magnitudes. Saved interval calibration is not revalidated for modified inputs. Test months are developmental, dependent and already inspected. Fixed dates are an economical diagnostic, not exhaustive attribution.',
            '', 'Evidence: protocol.json, monthly.csv, summary.json, examples.json, audit.json. Exact frozen checkpoint paths/hashes in audit.json. Reproduce: python run.py campaigns/champion_sensitivity_20261017/analyze.py on research host.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write('audit.json',dict(passed=True,folds=audits,checkpoint_sha256=checkpoints,
                           script_sha256=sha(Path(__file__)),training=False,protocol_sha256=sha(OUT/'protocol.json')))
    print('CHAMPION SENSITIVITY AUDIT PASSED',flush=True)

if __name__=='__main__': main()
