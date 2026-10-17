"""Restyle the unchanged saved example to the user's neon SVG reference."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-quantile-examples')
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/illustrative_quantile_bands_20261017'
GREEN='#25FF00';RED='#FF3854';BLUE='#28A9E2';YELLOW='#FFD43B'

def main():
 source=OUT/'example_predictions.csv';rows=list(csv.DictReader(source.open()));x=np.array([(int(r['hour_index'])-1)+(int(r['quarter_index'])-.5)/4 for r in rows]);y=np.array([float(r['observed']) for r in rows]);median=np.array([float(r['q0.5']) for r in rows]);lo=np.array([float(r['lower90']) for r in rows]);hi=np.array([float(r['upper90']) for r in rows]);assert len(rows)==96 and np.all(lo<=hi)
 plt.rcParams.update({'font.family':'Arial','font.weight':'bold','svg.fonttype':'none','text.color':GREEN,'axes.labelcolor':GREEN,'xtick.color':GREEN,'ytick.color':GREEN,'axes.edgecolor':GREEN,'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black'})
 fig=plt.figure(figsize=(19.2,10.8),dpi=100);ax=fig.add_axes([150/1920,190/1080,1650/1920,805/1080]);ax.set_facecolor('black')
 ax.fill_between(x,lo,hi,color=GREEN,alpha=.13,linewidth=0)
 ax.plot(x,lo,color=GREEN,alpha=.72,lw=2.16);ax.plot(x,hi,color=GREEN,alpha=.72,lw=2.16)
 ax.plot(x,median,color=RED,lw=4.32,solid_joinstyle='round',label='MODEL MEDIAN');ax.plot(x,y,color=BLUE,lw=3.6,solid_joinstyle='round',label='OBSERVED PRICE')
 ax.set_xlim(0,24);ax.set_ylim(-20,80);ax.set_xticks(range(0,25,3),[f'{h:02d}:00' for h in range(0,25,3)]);ax.set_yticks(range(-20,81,20));ax.tick_params(axis='both',labelsize=21,length=9,width=2.9,pad=11);ax.spines[['top','right']].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_linewidth(2.9)
 ax.set_ylabel('PRICE (€/MWh)',fontsize=24,labelpad=25,fontweight='bold')
 props=dict(arrowstyle='-|>',color=YELLOW,lw=3.6,mutation_scale=23,connectionstyle='arc3,rad=0')
 ax.annotate('CALM PERIOD =\nNARROWER BANDS',xy=(13.25,8),xytext=(9.7,58),color=YELLOW,fontsize=22,fontweight='bold',ha='left',va='center',arrowprops=props)
 ax.annotate('QUARTER-HOUR OSCILLATIONS =\nWIDER BANDS',xy=(22.6,32),xytext=(14.8,69),color=YELLOW,fontsize=22,fontweight='bold',ha='left',va='center',arrowprops=props)
 fig.text(.5,.962,'INTERVALS SHOW WHEN THE MODEL IS LESS CERTAIN',ha='center',va='center',fontsize=24.5,color=GREEN,fontweight='bold')
 handles=[Line2D([0],[0],color=BLUE,lw=5),Line2D([0],[0],color=RED,lw=5),Patch(facecolor=GREEN,edgecolor=GREEN,alpha=.4,linewidth=2)]
 legend=fig.legend(handles,['OBSERVED PRICE','MODEL MEDIAN','90% CONFORMAL INTERVAL'],loc='lower center',bbox_to_anchor=(.51,.045),ncol=3,frameon=False,fontsize=21.5,handlelength=2.5,columnspacing=2.0,labelcolor=GREEN)
 fig.text(.5,.018,'11 JUNE 2026 · ZONE 3 · ORIGINAL CHAMPION · UNCHANGED DATA / BANDS · ILLUSTRATIVE EXAMPLE',ha='center',color=GREEN,fontsize=12.5,fontweight='bold')
 fig.savefig(OUT/'calm_vs_volatile_neon.svg');fig.savefig(OUT/'calm_vs_volatile_neon.png',dpi=100);plt.close(fig)
 reference=Path('/Users/miguelnogales/Downloads/conformal_prediction_it.svg')
 receipt=dict(source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_svg_sha256=hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None,width_px=1920,height_px=1080,background='#000000',axis_band_color=GREEN,observed_color=BLUE,median_color=RED,annotation_color=YELLOW,font='Arial (reference specifies Lato with Arial fallback)',unchanged_data=True,displayed_band='Original saved calibrated90% interval; omitted raw quantile lines/50% fill to match single-band reference style')
 (OUT/'neon_style_receipt.json').write_text(json.dumps(receipt,indent=2));print('Saved editable SVG and 1920×1080 PNG; source values unchanged.')
if __name__=='__main__':main()
