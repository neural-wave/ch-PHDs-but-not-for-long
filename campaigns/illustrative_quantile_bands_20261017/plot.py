"""Illustrative saved example only: no fitting, inference or quantile modification."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/illustrative_quantile_bands_20261017'
DATE='2026-06-11';ZONE=3

def main():
 OUT.mkdir(parents=True,exist_ok=True);source=ROOT/'results/minimize_expanding_20261017/folds/2026-06/selected/fixed_blend_ratio/predictions.npz';z=np.load(source);i=int(np.flatnonzero(z['dates']==DATE)[0]);q=z['quantiles'][i,ZONE].reshape(96,11);y=z['observed'][i,ZONE].ravel();lo=z['lower_90'][i,ZONE].ravel();hi=z['upper_90'][i,ZONE].ravel();x=(np.arange(96)+.5)/4
 assert np.isfinite(q).all() and np.diff(q,axis=-1).min()>=-1e-9
 rough=abs(np.diff(y.reshape(24,4),n=2,axis=-1)).mean(-1);window=np.convolve(rough,np.ones(3)/3,'valid');starts=[int(window.argmin()),int(window.argmax())];colors=['#539875','#c18b35'];labels=['Calm stretch','Volatile / chainsaw stretch']
 fig=plt.figure(figsize=(14,8));gs=GridSpec(2,2,figure=fig,height_ratios=[1.4,1]);top=fig.add_subplot(gs[0,:]);bottom=[fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])];axes=[top]+bottom
 for n,ax in enumerate(axes):
  mask=np.ones(96,dtype=bool) if n==0 else (x>=starts[n-1])&(x<starts[n-1]+3)
  xx=x[mask];ax.fill_between(xx,lo[mask],hi[mask],color='#a4c5ed',alpha=.65,label='Calibrated 90% band');ax.fill_between(xx,q[mask,4],q[mask,6],color='#4789ce',alpha=.45,label='Raw q25–q75')
  for k in [0,1,2,3,7,8,9,10]:ax.plot(xx,q[mask,k],color='#739bca',lw=.5,alpha=.4)
  ax.plot(xx,q[mask,5],color='#1359a0',lw=1.6,label='q50 prediction');ax.plot(xx,y[mask],color='#d65b26',lw=1.3,label='Observed price');ax.grid(alpha=.2);ax.set_ylabel('EUR/MWh');ax.set_xlabel('Hour of delivery day')
  if n:
   width=float((hi[mask]-lo[mask]).mean());ax.set_title(f'{labels[n-1]}: hours {starts[n-1]}–{starts[n-1]+3}\nMean 90% width: {width:.2f} EUR/MWh',color=colors[n-1],loc='left')
 # Both zooms share the same price scale.
 vals=np.concatenate([lo[(x>=st)&(x<st+3)] for st in starts]+[hi[(x>=st)&(x<st+3)] for st in starts]+[y[(x>=st)&(x<st+3)] for st in starts]);pad=(vals.max()-vals.min())*.07
 for ax in bottom:ax.set_ylim(vals.min()-pad,vals.max()+pad)
 for st,c in zip(starts,colors):top.axvspan(st,st+3,color=c,alpha=.10)
 top.legend(loc='upper left',fontsize=8,ncol=4);top.set_xticks(range(0,25,2));coverage=float(((y>=lo)&(y<=hi)).mean());top.set_title(f'{DATE} · ZONE_{ZONE} · Original champion, unchanged quantiles\nFull-day empirical 90% coverage: {coverage:.1%}',loc='left')
 fig.suptitle('Narrower bands in calm periods, wider bands around quarter-hour oscillations',fontsize=14);fig.text(.5,.005,'Illustrative day and zooms selected after inspecting observations; not a new model or a general coverage claim.',ha='center',fontsize=9,color='#555555');fig.tight_layout(rect=[0,.025,1,.96]);fig.savefig(OUT/'calm_vs_volatile.png',dpi=180);fig.savefig(OUT/'calm_vs_volatile.svg');plt.close(fig)
 rows=[]
 for st,label in zip(starts,labels):
  mask=(x>=st)&(x<st+3);rows.append(dict(label=label,start_hour=st,end_hour=st+3,width90=float((hi[mask]-lo[mask]).mean()),observed_roughness=float(rough[st:st+3].mean())))
 summary=dict(model='minimize_expanding_20261017/fixed_blend_ratio',date=DATE,zone=ZONE,coverage90=coverage,examples=rows,quantiles_modified=False,model_refitted=False,selection='Posthoc illustrative selection using observed roughness/coverage/width association; zooms minimize/maximize observed3hour mean within-hour absolute second differences',source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(dict(hour_index=np.arange(96)//4+1,quarter_index=np.arange(96)%4+1,observed=y,lower90=lo,upper90=hi,**{f'q{tau:g}':q[:,n] for n,tau in enumerate(z['taus'])})).to_csv(OUT/'example_predictions.csv',index=False);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
