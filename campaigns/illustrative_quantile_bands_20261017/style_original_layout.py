"""Original June11 data/layout with palette-only reference styling."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-quantile-examples')
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/illustrative_quantile_bands_20261017'
GREEN='#25FF00';RED='#FF3854';BLUE='#28A9E2';YELLOW='#FFD43B'

def main():
 source=OUT/'example_predictions.csv';rows=list(csv.DictReader(source.open()));x=np.array([int(r['hour_index'])-1+(int(r['quarter_index'])-.5)/4 for r in rows]);q=np.array([[float(r['q'+str(t)]) for t in [.025,.05,.1,.2,.25,.5,.75,.8,.9,.95,.975]] for r in rows]);y=np.array([float(r['observed']) for r in rows]);lo=np.array([float(r['lower90']) for r in rows]);hi=np.array([float(r['upper90']) for r in rows]);assert len(rows)==96
 plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':GREEN,'axes.labelcolor':GREEN,'axes.edgecolor':GREEN,'xtick.color':GREEN,'ytick.color':GREEN})
 fig=plt.figure(figsize=(14,8));gs=GridSpec(2,2,figure=fig,height_ratios=[1.4,1]);top=fig.add_subplot(gs[0,:]);bottom=[fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])];axes=[top]+bottom;starts=[12,21];labels=['Calm stretch','Volatile / chainsaw stretch'];colors=[GREEN,YELLOW]
 for n,ax in enumerate(axes):
  mask=np.ones(96,dtype=bool) if n==0 else (x>=starts[n-1])&(x<starts[n-1]+3)
  xx=x[mask];ax.fill_between(xx,lo[mask],hi[mask],color=GREEN,alpha=.22,linewidth=0,label='Calibrated 90% band');ax.fill_between(xx,q[mask,4],q[mask,6],color=GREEN,alpha=.4,linewidth=0,label='Raw q25–q75')
  ax.plot(xx,lo[mask],color=GREEN,lw=.8,alpha=.8);ax.plot(xx,hi[mask],color=GREEN,lw=.8,alpha=.8)
  for k in [0,1,2,3,7,8,9,10]:ax.plot(xx,q[mask,k],color=GREEN,lw=.45,alpha=.25)
  ax.plot(xx,q[mask,5],color=RED,lw=1.8,label='q50 prediction');ax.plot(xx,y[mask],color=BLUE,lw=1.4,label='Observed price');ax.grid(color=GREEN,alpha=.10,lw=.5);ax.set_ylabel('EUR/MWh');ax.set_xlabel('Hour of delivery day');ax.spines[['top','right']].set_visible(False)
  for side in ['left','bottom']:ax.spines[side].set_linewidth(1.2)
  if n:
   width=float((hi[mask]-lo[mask]).mean());ax.set_title(f'{labels[n-1]}: hours {starts[n-1]}–{starts[n-1]+3}\nMean 90% width: {width:.2f} EUR/MWh',color=colors[n-1],loc='left',fontweight='bold')
   # Width marker touches the actual saved bounds; no adjusted interval.
   k=np.flatnonzero(mask)[-2];ax.annotate('',xy=(x[k],hi[k]),xytext=(x[k],lo[k]),arrowprops=dict(arrowstyle='<->',color=YELLOW,lw=1.6,mutation_scale=12))
 vals=np.concatenate([lo[(x>=st)&(x<st+3)] for st in starts]+[hi[(x>=st)&(x<st+3)] for st in starts]+[y[(x>=st)&(x<st+3)] for st in starts]);pad=(vals.max()-vals.min())*.07
 for ax in bottom:ax.set_ylim(vals.min()-pad,vals.max()+pad)
 for st,c in zip(starts,colors):top.axvspan(st,st+3,color=c,alpha=.055)
 top.legend(loc='upper left',fontsize=8,ncol=4,facecolor='black',edgecolor=GREEN,labelcolor=GREEN);top.set_xticks(range(0,25,2));coverage=float(((y>=lo)&(y<=hi)).mean());top.set_title(f'2026-06-11 · ZONE_3 · Original champion\nFull-day empirical 90% coverage: {coverage:.1%}',loc='left',fontweight='bold')
 fig.suptitle('Narrower bands in calm periods, wider bands around quarter-hour oscillations',fontsize=14,color=GREEN,fontweight='bold');fig.text(.5,.005,'Illustrative day and zooms selected after inspecting observations. Original predictions and interval bounds unchanged.',ha='center',fontsize=9,color=GREEN,alpha=.8);fig.tight_layout(rect=[0,.025,1,.96]);fig.savefig(OUT/'calm_vs_volatile_original_layout_neon.png',dpi=180);fig.savefig(OUT/'calm_vs_volatile_original_layout_neon.svg');plt.close(fig)
 (OUT/'original_layout_style_receipt.json').write_text(json.dumps(dict(source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),date='2026-06-11',zone=3,layout='Original full-day curve plus two same-axis three-hour zooms',changed='Palette/font/line/fill styling and truthful bound-to-bound width markers',data_modified=False,calibration_modified=False),indent=2));print('Original layout and saved bands preserved.')
if __name__=='__main__':main()
