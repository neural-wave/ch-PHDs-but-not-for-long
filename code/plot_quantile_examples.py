import os
os.environ['MPLCONFIGDIR']='/private/tmp/mpl-quantile-examples'
import tarfile,io,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; out=root/'results/quantile_examples_20261017'
items=[]
with tarfile.open(root/'results/minimize_expanding_20261017_completed.tar.gz') as t:
 for m in t:
  if '/folds/' in m.name and m.name.endswith('/selected/fixed_blend_ratio/predictions.npz'):
   with np.load(io.BytesIO(t.extractfile(m).read())) as z:
    items.append({k:z[k].copy() for k in z.files})
items.sort(key=lambda z:str(z['dates'][0]));print([(str(z['dates'][0]),z['quantiles'].shape) for z in items],flush=True)
q=np.concatenate([z['quantiles'] for z in items]); y=np.concatenate([z['observed'] for z in items]);lo=np.concatenate([z['lower_90'] for z in items]);hi=np.concatenate([z['upper_90'] for z in items]);dates=np.concatenate([z['dates'] for z in items]); pct=np.concatenate([z['uncertainty_percentile'] for z in items]);w=hi-lo
assert q.shape==y.shape+(11,); assert np.isfinite(q).all() and np.diff(q,axis=-1).min()>=-1e-8
# Select entire day/zone curves using only mean calibrated interval width.
dw=w.mean(axis=(2,3)); chosen=[np.unravel_index(np.argmin(dw),dw.shape),np.unravel_index(np.argmax(dw),dw.shape)]
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True); rows=[]
for ax,(d,z),label in zip(axes,chosen,['Narrowest day/zone bands','Widest day/zone bands']):
 x=np.arange(96)/4; a=q[d,z].reshape(96,11); obs=y[d,z].ravel(); l=lo[d,z].ravel();u=hi[d,z].ravel()
 ax.fill_between(x,l,u,color='#a4c5ed',alpha=.55,label='Calibrated 90% interval')
 ax.fill_between(x,a[:,4],a[:,6],color='#4789ce',alpha=.45,label='Raw q25–q75')
 for k in [0,1,2,3,7,8,9,10]: ax.plot(x,a[:,k],color='#739bca',lw=.6,alpha=.55)
 ax.plot(x,a[:,5],color='#1359a0',lw=1.5,label='q50 prediction');ax.plot(x,obs,color='#d65b26',lw=1.2,label='Observed price')
 mae=float(abs(a[:,5]-obs).mean());width=float((u-l).mean());coverage=float(((obs>=l)&(obs<=u)).mean())
 ax.set_title(f'{label}: {dates[d]} · ZONE_{z} | mean 90% width {width:.2f} · MAE {mae:.2f} EUR/MWh',loc='left');ax.set_ylabel('EUR/MWh');ax.grid(alpha=.2)
 rows.append(dict(label=label,date=str(dates[d]),zone=f'ZONE_{z}',mean_width90=width,mae=mae,coverage90=coverage,mean_raw_q90_q10=float((a[:,8]-a[:,2]).mean())))
axes[0].legend(loc='upper left',ncol=4,fontsize=8);axes[-1].set_xlabel('Hour of delivery day (hour_index 1 starts at 0)');axes[-1].set_xticks(range(0,25,2));fig.suptitle('Research champion: quantile dispersion contrasts\nSelected by interval width alone; developmental February–September backtests',fontsize=13);fig.tight_layout();fig.savefig(out/'confidence_examples.png',dpi=170);fig.savefig(out/'confidence_examples.svg');plt.close(fig)
with (out/'selected_predictions.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['date','zone','hour_index','quarter_index','observed']+[f'q{t}' for t in items[0]['taus']]+['lower90','upper90','uncertainty_percentile'])
 for d,z in chosen:
  for h in range(24):
   for k in range(4): writer.writerow([str(dates[d]),f'ZONE_{z}',h+1,k+1,y[d,z,h,k],*q[d,z,h,k],lo[d,z,h,k],hi[d,z,h,k],pct[d,z,h,k]])
summary={'model':'minimize_expanding_20261017/fixed_blend_ratio','selection':'Minimum/maximum mean calibrated 90% width across all 964 held-out day-zone curves; observed errors not used to select','examples':rows,'bins':{}}
err=abs(q[...,5]-y)
for name,mask in [('narrow',pct<=.25),('middle',(pct>.25)&(pct<.75)),('wide',pct>=.75)]: summary['bins'][name]={'n':int(mask.sum()),'mae':float(err[mask].mean()),'mean_width90':float(w[mask].mean()),'coverage90':float(((y>=lo)&(y<=hi))[mask].mean())}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
