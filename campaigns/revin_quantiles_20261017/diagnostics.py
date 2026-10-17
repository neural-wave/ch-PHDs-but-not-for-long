"""Paired scale-adjusted uncertainty diagnostics for fitted models; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train as t

def corr(a,b):
    a=pd.Series(np.asarray(a).ravel()).rank();b=pd.Series(np.asarray(b).ravel()).rank()
    return float(a.corr(b)) if len(a)>1 and a.std()>0 and b.std()>0 else None

def main():
    root=t.ROOT;bins=[];days=[];quarters=[];examples={};affines=[]
    for folder in sorted((root/'folds').iterdir()):
        d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);start=nt+ns+nc
        scale=np.sqrt(d['h'].var(-1)+1);cal_scale=scale[nt+ns:start];test_scale=scale[start:]
        cal_y=d['y'][nt+ns:start];truth=d['y'][start:];selection=json.loads((folder/'selection.json').read_text())
        for recipe in t.RECIPES:
            row=selection['by_recipe'][recipe];trial=folder/'trials'/str(row['config']['id']);allq=np.load(trial/'quantiles.npy');calq=allq[ns:ns+nc]
            bands=t.evaluation.intervals(calq,cal_y,calq)[90];calw=(bands['upper']-bands['lower'])/cal_scale[:,:,None,None]
            z=np.load(folder/'selected'/recipe/'predictions.npz');q=z['quantiles'];width=(z['upper_90']-z['lower_90'])/test_scale[:,:,None,None];err=abs(q[...,5]-truth)/test_scale[:,:,None,None]
            baseerr=abs(d['b'][start:]-truth)/test_scale[:,:,None,None];covered=(truth>=z['lower_90'])&(truth<=z['upper_90'])
            pct=np.empty_like(width)
            for zone in range(4):
                cuts=np.quantile(cal_scale[:,zone],[.25,.75])
                stratum=np.where(test_scale[:,zone]<=cuts[0],0,np.where(test_scale[:,zone]>=cuts[1],2,1))
                for k in range(4):
                    ref=np.sort(calw[:,zone,:,k].ravel());v=width[:,zone,:,k]
                    pct[:,zone,:,k]=(np.searchsorted(ref,v,'left')+np.searchsorted(ref,v,'right'))/(2*len(ref))
                for band in range(3):
                    ids=stratum==band
                    if ids.any():bins.append(dict(month=folder.name,recipe=recipe,zone=zone,scale_stratum=band,n=int(ids.sum()*96),spearman_normalized_width_error=corr(width[ids,zone],err[ids,zone])))
            for label,mask in [('narrow',pct<=.25),('middle',(pct>.25)&(pct<.75)),('wide',pct>=.75)]:
                quarters.append(dict(month=folder.name,recipe=recipe,label=label,n=int(mask.sum()),normalized_error_sum=float(err[mask].sum()),normalized_width_sum=float(width[mask].sum()),normalized_baseline_error_sum=float(baseerr[mask].sum()),covered=int(covered[mask].sum())))
            for i,date in enumerate(z['dates']):
                for zone in range(4):
                    days.append(dict(date=str(date),recipe=recipe,zone=zone,scale=float(test_scale[i,zone]),normalized_width90=float(width[i,zone].mean()),normalized_mae=float(err[i,zone].mean()),mae=float(abs(q[i,zone,...,5]-truth[i,zone]).mean()),coverage90=float(covered[i,zone].mean())))
                    if (str(date),zone) in [('2026-04-26',3),('2026-03-14',2)]:
                        examples[str(date),zone,recipe]=dict(q=q[i,zone],y=truth[i,zone],lo=z['lower_90'][i,zone],hi=z['upper_90'][i,zone],mu=d['h'][start+i,zone].mean(),s=test_scale[i,zone])
            ck=t.torch.load(trial/'model.pt',weights_only=False,map_location='cpu')['state']
            affines.append(dict(month=folder.name,recipe=recipe,gamma=ck['affine_log_scale'].clamp(-3,3).exp().tolist(),beta=ck['affine_bias'].tolist()))
    pd.DataFrame(days).to_csv(root/'day_uncertainty.csv',index=False);pd.DataFrame(bins).to_csv(root/'within_scale_associations.csv',index=False)
    df=pd.DataFrame(quarters);df.to_csv(root/'normalized_confidence_monthly.csv',index=False)
    grouped=df.groupby(['recipe','label'])[['n','normalized_error_sum','normalized_width_sum','normalized_baseline_error_sum','covered']].sum()
    grouped['normalized_mae']=grouped.normalized_error_sum/grouped.n;grouped['normalized_width90']=grouped.normalized_width_sum/grouped.n;grouped['coverage90']=grouped.covered/grouped.n
    grouped['normalized_error_ratio']=grouped.normalized_error_sum/grouped.normalized_baseline_error_sum;grouped.to_csv(root/'normalized_confidence_pooled.csv')
    assert all(sum(v['n'] for v in quarters if v['recipe']==recipe)==92544 for recipe in t.RECIPES)
    (root/'selected_affine.json').write_text(json.dumps(affines,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,sharey=True);exports=[]
    for i,(date,zone) in enumerate([('2026-03-14',2),('2026-04-26',3)]):
        for j,recipe in enumerate(t.RECIPES):
            e=examples[date,zone,recipe];q=e['q'].reshape(96,11);y=e['y'].ravel();lo=e['lo'].ravel();hi=e['hi'].ravel();norm=lambda a:(a-e['mu'])/e['s'];x=(np.arange(96)+.5)/4;ax=axes[i,j]
            ax.fill_between(x,norm(lo),norm(hi),alpha=.45,color='#a4c5ed',label='Calibrated90%');ax.fill_between(x,norm(q[:,4]),norm(q[:,6]),alpha=.45,color='#4789ce',label='q25–q75');ax.plot(x,norm(q[:,5]),color='#1359a0',label='q50');ax.plot(x,norm(y),color='#d65b26',label='Observed')
            ax.set_title(f'{date} ZONE_{zone} · {recipe}\nnormalized width {(hi-lo).mean()/e["s"]:.3f} · MAE {abs(q[:,5]-y).mean()/e["s"]:.3f}',fontsize=10);ax.grid(alpha=.2)
            for k in range(96):exports.append(dict(date=date,zone=zone,recipe=recipe,hour=k//4+1,quarter=k%4+1,observed=y[k],lower90=lo[k],upper90=hi[k],**{f'q{tau:g}':q[k,n] for n,tau in enumerate(t.TAUS)}))
    axes[0,0].legend(fontsize=8,ncol=2);fig.supxlabel('Hour of delivery day');fig.supylabel('(price − known hourly daily mean) / daily scale');fig.suptitle('Matched trained fixed-scale versus learned-affine RevIN quantile CNN\nShared normalized axes; previously inspected developmental examples');fig.tight_layout();fig.savefig(root/'uncertainty_examples.png',dpi=170);fig.savefig(root/'uncertainty_examples.svg');plt.close(fig);pd.DataFrame(exports).to_csv(root/'example_predictions.csv',index=False)
    lines=['# Scale-adjusted uncertainty comparison','', 'Both models were trained fresh; RevIN reverses input-only daily scale and its learned positive affine on output deviations. Quantile plots use shared daily-normalized axes. Example dates were previously inspected, so do not treat them as independent confirmation.','', 'Normalized narrow/middle/wide thresholds come from preceding calibration outputs within zone/quarter. Pooled values are target-weighted diagnostic quantities, separate from the primary equal-month MAE/interpolation score. Scaling errors inversely by daily volatility here does not change the original-currency training loss.','', grouped[['n','normalized_mae','normalized_width90','coverage90','normalized_error_ratio']].to_string(),'', 'within_scale_associations.csv gives rank association within fold, zone and low/middle/high daily-scale groups defined from calibration quartiles. These are descriptive dependent-data diagnostics; no test thresholds or models are selected. Check consistency across these strata rather than interpreting pooled association alone. No novelty/rarity mechanism or deployment promotion is established.','', 'See REPORT.md, audit.json, selected_affine.json, day_uncertainty.csv, normalized_confidence_monthly.csv, normalized_confidence_pooled.csv and uncertainty_examples.png/svg.']
    (root/'UNCERTAINTY.md').write_text('\n'.join(lines)+'\n');print('UNCERTAINTY DIAGNOSTICS COMPLETE',flush=True)
if __name__=='__main__':main()
