"""Unattended post-fit diagnostic: predicted sawtooth shape versus marginal width."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/revin_quantiles_20261017'
OUT=ROOT/'results/revin_chainsaw_20261017'

def roughness(a):
    # Second differences are zero for an exactly linear four-quarter segment.
    return np.abs(np.diff(a,n=2,axis=-1)).mean(-1)

def main(wait_pid=None):
    if wait_pid:
        start=time.monotonic()
        while not (SOURCE/'completion.json').exists():
            if time.monotonic()-start>1800:raise RuntimeError('Training dependency did not complete within30minutes')
            try:os.kill(wait_pid,0)
            except ProcessLookupError:raise RuntimeError('Training worker ended without successful completion receipt')
            time.sleep(10)
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    OUT.mkdir(parents=True,exist_ok=True);allrows=[];examples={};associations=[];receipt={}
    for folder in sorted((SOURCE/'folds').iterdir()):
        d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc;s=np.sqrt(d['h'].var(-1)+1);selection=json.loads((folder/'selection.json').read_text())
        for recipe in ['compact_fixed','compact_day_reversible']:
            path=folder/'selected'/recipe/'predictions.npz';receipt[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();z=np.load(path)
            q=z['quantiles'];pred=q[...,5];y=z['observed'];scale=s[start:];predrough=roughness(pred)/scale[:,:,None];obsrough=roughness(y)/scale[:,:,None];width=(z['upper_90']-z['lower_90']).mean(-1)/scale[:,:,None];raw_width=(q[...,9]-q[...,1]).mean(-1)/scale[:,:,None];error=abs(pred-y).mean(-1)/scale[:,:,None]
            trial=folder/'trials'/str(selection['by_recipe'][recipe]['config']['id'])/'quantiles.npy';cal=np.load(trial)[ns:ns+nc];calrough=roughness(cal[...,5])/s[nt+ns:start,:,None]
            for zone in range(4):
                cuts=np.quantile(calrough[:,zone],[.25,.75]);labels=np.where(predrough[:,zone]<=cuts[0],'smooth',np.where(predrough[:,zone]>=cuts[1],'chainsaw','middle'))
                for i,date in enumerate(z['dates']):
                    for h in range(24):allrows.append(dict(month=folder.name,date=str(date),zone=zone,hour=h+1,recipe=recipe,predicted_shape=labels[i,h],predicted_roughness=predrough[i,zone,h],observed_roughness=obsrough[i,zone,h],normalized_width90=width[i,zone,h],normalized_raw_width90=raw_width[i,zone,h],normalized_mae=error[i,zone,h]))
                # Each stratum has one fixed daily scale: association across its24hours.
                for i,date in enumerate(z['dates']):
                    a=pd.Series(predrough[i,zone]).rank();b=pd.Series(width[i,zone]).rank();c=float(a.corr(b)) if a.std()>0 and b.std()>0 else None
                    associations.append(dict(date=str(date),zone=zone,recipe=recipe,within_day_spearman=c))
                # Selection by predicted shape contrast only; observed outputs are not used.
                contrasts=predrough[:,zone].max(-1)-predrough[:,zone].min(-1);i=int(np.argmax(contrasts));key=recipe
                if key not in examples or contrasts[i]>examples[key]['contrast']:
                    examples[key]=dict(date=str(z['dates'][i]),zone=zone,contrast=float(contrasts[i]),q=q[i,zone],y=y[i,zone],lo=z['lower_90'][i,zone],hi=z['upper_90'][i,zone],scale=float(scale[i,zone]),mu=float(d['h'][start+i,zone].mean()),rough=predrough[i,zone])
    df=pd.DataFrame(allrows);assert len(df)==46272;assert np.isfinite(df.select_dtypes('number')).all().all();df.to_csv(OUT/'hour_shapes.csv',index=False)
    summary=df.groupby(['recipe','predicted_shape']).agg(hours=('hour','size'),predicted_roughness=('predicted_roughness','mean'),observed_roughness=('observed_roughness','mean'),normalized_width90=('normalized_width90','mean'),normalized_raw_width90=('normalized_raw_width90','mean'),normalized_mae=('normalized_mae','mean'));summary.to_csv(OUT/'shape_summary.csv')
    a=pd.DataFrame(associations);a.to_csv(OUT/'within_day_associations.csv',index=False)
    association_summary={recipe:dict(mean_within_day_rank_correlation=float(g.within_day_spearman.mean()),fraction_positive=float((g.within_day_spearman.dropna()>0).mean()),valid_day_zones=int(g.within_day_spearman.notna().sum())) for recipe,g in a.groupby('recipe')}
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True);exports=[]
    for row,(recipe,e) in enumerate(examples.items()):
        for col,h in enumerate([int(e['rough'].argmin()),int(e['rough'].argmax())]):
            ax=axes[row,col];q=e['q'][h];y=e['y'][h];lo=e['lo'][h];hi=e['hi'][h];norm=lambda a:(a-e['mu'])/e['scale'];x=np.arange(1,5)
            ax.fill_between(x,norm(lo),norm(hi),alpha=.45,color='#a4c5ed',label='Calibrated90%');ax.fill_between(x,norm(q[:,4]),norm(q[:,6]),alpha=.45,color='#4789ce',label='q25–q75');ax.plot(x,norm(q[:,5]),'-o',color='#1359a0',label='q50');ax.plot(x,norm(y),'-o',color='#d65b26',label='Observed');ax.grid(alpha=.2)
            ax.set_title(f'{recipe}\n{e["date"]} ZONE_{e["zone"]} hour{h+1} · '+('smoothest' if col==0 else 'most jagged')+f'\nroughness{e["rough"][h]:.3f} · width{(hi-lo).mean()/e["scale"]:.3f}',fontsize=9)
            exports.append(dict(recipe=recipe,date=e['date'],zone=e['zone'],hour=h+1,predicted_roughness=float(e['rough'][h]),normalized_width90=float((hi-lo).mean()/e['scale'])))
    axes[0,0].legend(fontsize=8);fig.suptitle('Does predicted sawtooth shape coincide with wider marginal quantiles?\nEach row compares two hours in the SAME day/zone; selected by predicted roughness alone');fig.supxlabel('Quarter within hour');fig.supylabel('(price − daily hourly mean) / input daily scale');fig.tight_layout();fig.savefig(OUT/'smooth_vs_chainsaw.png',dpi=180);fig.savefig(OUT/'smooth_vs_chainsaw.svg');plt.close(fig)
    (OUT/'examples.json').write_text(json.dumps(exports,indent=2));(OUT/'associations.json').write_text(json.dumps(association_summary,indent=2))
    (OUT/'manifest.json').write_text(json.dumps(dict(source_job='1e1315765ed2bc28e68bec9c',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prediction_sha256=receipt),indent=2))
    text=['# Predicted shape versus quantile dispersion','', 'Requested diagnostic: bands should spread for predicted chainsaw quarter-hour shapes and narrow for predicted smooth shapes. This analysis tests that behavior; it does not impose it on the model.','', 'Roughness is mean absolute second difference of the four q50 prices within each hour, divided by input-only daily scale sqrt(hourly variance+1). It is zero for a linear segment. Smooth/chainsaw are descriptive lower/upper-quartile roughness labels fixed from preceding calibration q50 predictions per zone, not a risk classifier. Shape here means predicted median; observed shape is a separate diagnostic.','',summary.to_string(),'',json.dumps(association_summary,indent=2),'','within_day_associations.csv measures roughness/width association across24hours with identical day/zone scale, directly addressing scale confounding. Interpret alongside raw90 widths, calibration effects and observed error. The example panels pick the day/zone with greatest predicted roughness contrast, then its smoothest and most jagged hours. No observed error selects examples. These are exploratory extrema in developmental folds.','', 'Marginal quantile bands measure dispersion of each quarter, not joint trajectory oscillation or unfamiliarity. A confidently predicted chainsaw can have narrow bands; a smooth median can hide uncertain alternating trajectories. Learned-affine RevIN changes scaling and representation but does not guarantee a shape-width relationship. No such mechanism is claimed from the plots alone.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');(OUT/'completion.json').write_text(json.dumps(dict(passed=True,hours=len(df),source_audit_passed=True,examples=exports),indent=2));print('CHAINSAW DIAGNOSTIC COMPLETE',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args();main(args.wait_pid)
