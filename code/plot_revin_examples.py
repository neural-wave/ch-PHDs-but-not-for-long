"""Display saved champion examples in reversible day/zone coordinates; no refit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-quantile-examples')
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/quantile_examples_20261017'
def main():
 groups=defaultdict(list)
 for r in csv.DictReader((OUT/'selected_predictions.csv').open()):groups[r['date'],r['zone']].append(r)
 fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,sharey=True);summary=[]
 for ax,((date,zone),rows) in zip(axes,groups.items()):
  y=np.array([float(r['observed']) for r in rows]).reshape(24,4);h=y.mean(1);mu=h.mean();scale=np.sqrt(h.var()+1)
  q=np.array([[float(r['q'+str(t)]) for t in [.025,.05,.1,.2,.25,.5,.75,.8,.9,.95,.975]] for r in rows]);lo=np.array([float(r['lower90']) for r in rows]);hi=np.array([float(r['upper90']) for r in rows]);x=(np.arange(96)+.5)/4
  base=np.interp(x,np.arange(24)+.5,h).reshape(24,4);base+=h[:,None]-base.mean(1)[:,None]
  norm=lambda a:(a-mu)/scale
  assert np.max(abs(norm(q)*scale+mu-q))<1e-9
  ax.fill_between(x,norm(lo),norm(hi),color='#a4c5ed',alpha=.6,label='Calibrated 90%')
  ax.fill_between(x,norm(q[:,4]),norm(q[:,6]),color='#4789ce',alpha=.45,label='Raw q25–q75')
  for k in [0,1,2,3,7,8,9,10]:ax.plot(x,norm(q[:,k]),lw=.55,color='#739bca',alpha=.5)
  ax.plot(x,norm(q[:,5]),color='#1359a0',label='q50');ax.plot(x,norm(y.ravel()),color='#d65b26',label='Observed');ax.plot(x,norm(base.ravel()),color='#333333',ls='--',lw=.9,label='Interpolation')
  width=float((hi-lo).mean()/scale);mae=float(abs(q[:,5]-y.ravel()).mean()/scale);bmae=float(abs(base-y).mean()/scale)
  ax.set_title(f'{date} · {zone} | normalized width {width:.3f} · MAE {mae:.3f} · interpolation MAE {bmae:.3f}',loc='left');ax.grid(alpha=.2);ax.set_ylabel('(price − daily mean) / daily scale')
  summary.append(dict(date=date,zone=zone,mean=float(mu),scale=float(scale),normalized_width90=width,normalized_mae=mae,normalized_interpolation_mae=bmae,mae_ratio=mae/bmae))
 axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Hour of delivery day');axes[-1].set_xticks(range(0,25,2));fig.suptitle('Saved champion outputs in RevIN-style coordinates\nSame y-axis scale · display transform only, no retraining');fig.tight_layout();fig.savefig(OUT/'revin_coordinates.png',dpi=170);fig.savefig(OUT/'revin_coordinates.svg');plt.close(fig)
 (OUT/'revin_coordinates.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
