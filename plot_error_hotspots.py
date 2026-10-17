"""Descriptive September error diagnostics from frozen, August-selected predictions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/error_hotspots_20261017'
OUT.mkdir(exist_ok=True)
cache = np.load(ROOT/'results/monthly_joint_blend_gpu1/test_predictions.npz')
dates, y = cache['dates'], cache['observed']
official = np.load(ROOT/'results/simple_5090/data.npz')
assert np.array_equal(dates, official['dates'][-30:])
assert np.array_equal(y, official['y'][-30:])
sel = json.loads((ROOT/'results/simple_5090/selection.json').read_text())
pred = {'Joint blend': cache['predicted']}
for label, family in [('Linear', 'linear'), ('CNN', 'cnn')]:
    config = sel['by_family'][family]
    p = np.load(ROOT/f"results/simple_5090/trials/{config['config']['id']}/predictions.npy")
    assert p.shape == (61, 4, 24, 4)
    pred[label] = p[-30:]
    assert abs(np.abs(pred[label]-y).mean()-config['september_mae']) < 1e-9
pred['Interpolation'] = cache['baseline']
err = {k: np.abs(p-y) for k,p in pred.items()}
for p in pred.values():
    assert np.isfinite(p).all() and np.max(np.abs(p.mean(-1)-y.mean(-1))) < 1e-4
colors = dict(zip(pred, ['#007f86','#e78b25','#8a59a5','#9b9b9b']))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(13,4.2))
width=.19
for i,(name,e) in enumerate(err.items()):
    axes[0].bar(np.arange(4)+(i-1.5)*width,e.mean((0,2,3)),width,color=colors[name],label=f'{name} ({e.mean():.2f})')
    axes[1].plot(np.arange(1,31),e.mean((1,2,3)),color=colors[name],label=name,lw=1.7)
axes[0].set(xticks=range(4),xticklabels=[f'ZONE_{z}' for z in range(4)],ylabel='MAE (EUR/MWh)',title='Errors by anonymized zone')
axes[0].legend(title='Model (overall MAE)',fontsize=9)
axes[1].set(xlabel='September day',ylabel='MAE (EUR/MWh)',title='Hard days shared across models',xticks=[1,5,10,15,20,25,30])
for ax in axes: ax.grid(axis='y',alpha=.2)
fig.suptitle('Where the strongest models struggle · September 2026 development set')
fig.tight_layout();save(fig,'01_zone_and_day')

e = err['Joint blend']; heat = e.mean(-1)
fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
vmax=float(np.quantile(heat,.99))
for z,ax in enumerate(axes.flat):
    im=ax.imshow(heat[:,z,:],aspect='auto',origin='lower',extent=[0,24,.5,30.5],vmin=0,vmax=vmax,cmap='magma')
    ax.set(title=f'ZONE_{z} · mean {e[:,z].mean():.2f}',xlabel='Hour start (dataset index − 1)',ylabel='September day',xticks=[0,4,8,12,16,20,24])
fig.colorbar(im,ax=axes.ravel().tolist(),label=f'Hourly MAE (EUR/MWh); colors capped at {vmax:.1f}')
fig.suptitle('Joint blend: exact days and hours with large errors')
save(fig,'02_day_hour_heatmap')

fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for name,e1 in err.items():
    axes[0].plot(np.arange(96)/4,e1.mean((0,1)).reshape(96),label=name,color=colors[name])
    values=e1.mean((0,1)).reshape(96)
    for quarter,marker in [(0,'o'),(3,'s')]:
        axes[0].plot(np.arange(96)[quarter::4]/4,values[quarter::4],linestyle='none',marker=marker,markersize=3,markerfacecolor='white',markeredgecolor=colors[name])
    axes[1].plot(range(1,5),e1.mean((0,1,2)),'o-',color=colors[name])
hour_price = np.broadcast_to(y.mean(-1,keepdims=True),y.shape)
within_hour = np.broadcast_to(np.ptp(y,axis=-1,keepdims=True),y.shape)
edges=[0,5,10,20,40,np.inf]; bins=np.digitize(within_hour.ravel(),edges)-1
counts=[int((bins==i).sum()) for i in range(5)]
for name,e1 in err.items():
    vals=[float(e1.ravel()[bins==i].mean()) for i in range(5)]
    axes[2].plot(range(5),vals,'o-',color=colors[name])
axes[0].set(xlabel='Hour start (dataset index − 1)',ylabel='MAE (EUR/MWh)',title='Time of day',xticks=[0,4,8,12,16,20,24]);axes[0].legend(fontsize=9)
axes[1].set(xlabel='Quarter within hour',title='Which quarter is hardest?',xticks=[1,2,3,4],xticklabels=['Q1\nFirst\n:00–:15','Q2\n:15–:30','Q3\n:30–:45','Q4\nLast\n:45–:60'])
axes[2].set(title='Observed within-hour price range',xlabel='Range (EUR/MWh) · share of observations',xticks=range(5),xticklabels=[f'{a}\n{c/y.size:.0%}' for a,c in zip(['0–5','5–10','10–20','20–40','40+'],counts)])
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Error patterns · ○ first quarter (:00) · □ last quarter (:45)\nRange uses observed targets and is diagnostic only')
fig.tight_layout();save(fig,'03_time_and_volatility')

dayzone=err['Joint blend'].mean((2,3)); order=list(np.argsort(dayzone.ravel())[::-1][:3])
point_idx=np.unravel_index(e.argmax(),e.shape)
order.append(np.ravel_multi_index(point_idx[:2],dayzone.shape))
fig,axes=plt.subplots(4,1,figsize=(13,12),sharex=True)
worst=[]
for ax,idx in zip(axes,order):
    d,z=np.unravel_index(idx,dayzone.shape);x=np.arange(96)/4
    for hour in range(24):
        if hour % 2 == 0: ax.axvspan(hour,hour+1,color='#e9eef4',alpha=.65,zorder=0)
        ax.axvline(hour,color='#bdc5ce',lw=.5,alpha=.6,zorder=0)
    ax.plot(x,y[d,z].ravel(),color='black',lw=2,label='Observed')
    for name in ['Joint blend','Linear','CNN']:
        ax.plot(x,pred[name][d,z].ravel(),color=colors[name],lw=1.4,label=name)
    for values,color in [(y[d,z].ravel(),'black')]+[(pred[name][d,z].ravel(),colors[name]) for name in ['Joint blend','Linear','CNN']]:
        for quarter,marker in [(0,'o'),(3,'s')]:
            ax.plot(x[quarter::4],values[quarter::4],linestyle='none',marker=marker,markersize=4,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1,zorder=4)
    ax.fill_between(x,y[d,z].ravel(),pred['Joint blend'][d,z].ravel(),color=colors['Joint blend'],alpha=.12)
    ax.set(title=f'{dates[d]} · ZONE_{z} · joint-blend daily MAE {dayzone[d,z]:.2f} EUR/MWh',ylabel='Price (EUR/MWh)');ax.grid(alpha=.2)
    worst.append(dict(date=str(dates[d]),zone=f'ZONE_{z}',mae=float(dayzone[d,z])))
handles,labels=axes[0].get_legend_handles_labels()
handles += [Line2D([],[],color='black',marker='o',markerfacecolor='white',linestyle='none'),Line2D([],[],color='black',marker='s',markerfacecolor='white',linestyle='none')]
labels += ['First quarter (:00–:15)','Last quarter (:45–:60)']
axes[0].legend(handles,labels,ncol=3,fontsize=9)
axes[-1].set(xlabel='Quarter-hour start · alternating shading separates hours',xticks=np.arange(0,25,1),xticklabels=[f'{h:02d}:00' for h in range(25)])
axes[-1].tick_params(axis='x',labelrotation=45,labelsize=9)
fig.suptitle('Joint blend: three hardest day–zone cases, then the largest single error\n○ first quarter of each hour · □ last quarter of each hour')
fig.tight_layout();save(fig,'04_worst_examples')

flat=e.ravel(); maxidx=np.unravel_index(flat.argmax(),e.shape);d,z,h,q=maxidx
threshold=float(np.quantile(flat,.95)); top=flat>=threshold
summary={'period':'2026-09 development set; previously inspected, not blind','models':{},'worst_day_zones':worst,'maximum_point_error':dict(date=str(dates[d]),zone=f'ZONE_{z}',hour_index=int(h+1),quarter_index=int(q+1),absolute_error=float(e[maxidx]),observed=float(y[maxidx]),predicted=float(pred['Joint blend'][maxidx])),'worst_5pct_error_share':float(flat[top].sum()/flat.sum()),'top_5pct_threshold':threshold,'observed_range_bins':{'counts':counts,'joint_blend_mae':[float(flat[bins==i].mean()) for i in range(5)]}}
for name,e1 in err.items():
    summary['models'][name]={'mae':float(e1.mean()),'by_zone':e1.mean((0,2,3)).tolist(),'by_quarter':e1.mean((0,1,2)).tolist(),'by_hour':e1.mean((0,1,3)).tolist()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
