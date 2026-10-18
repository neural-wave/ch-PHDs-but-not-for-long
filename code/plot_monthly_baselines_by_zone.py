"""Monthly baseline MAE by anonymized market; requires matplotlib."""
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
methods=['hourly_mean','linear_interpolation','linear_interpolation_mean_corrected']
labels=['Repeat hourly mean','Linear interpolation','Mean-corrected interpolation']
colors=['#6b7280','#d97706','#007c91']
groups=defaultdict(list)
with (root/'results/baseline_daily_metrics.csv').open() as f:
    for r in csv.DictReader(f):
        groups[r['delivery_date'][:7],r['market_id'],r['method']].append(float(r['mae_eur_mwh']))
months=sorted({k[0] for k in groups})
zones=sorted({k[1] for k in groups})
assert len(months)==12 and len(zones)==4
rows=[]
for month in months:
    for zone in zones:
        row={'month':month,'market_id':zone,'days':len(groups[month,zone,methods[0]])}
        for method in methods:
            vals=groups[month,zone,method]
            assert len(vals)==row['days']
            row[method]=sum(vals)/len(vals)
        rows.append(row)
assert sum(r['days'] for r in rows)==363*4
with (root/'results/monthly_baseline_mae_by_zone.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Check the disaggregation reproduces the existing overall monthly results.
with (root/'results/monthly_baseline_mae.csv').open() as f:
    for original in csv.DictReader(f):
        selected=[r for r in rows if r['month']==original['month']]
        for method in methods:
            assert abs(sum(r[method] for r in selected)/4-float(original[method]))<1e-10
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
for ax,zone in zip(axes.flat,zones):
    selected=[r for r in rows if r['market_id']==zone]
    for method,label,color in zip(methods,labels,colors):
        ax.plot(range(12),[r[method] for r in selected],marker='o',markersize=4,linewidth=2,label=label,color=color)
    ax.set_title(zone,fontweight='bold',loc='left')
    ax.grid(axis='y',alpha=.2)
    ax.set_ylim(0, max(r[m] for r in rows for m in methods)*1.08)
    ax.set_xticks(range(12),['Oct\n2025','Nov','Dec','Jan\n2026','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'])
    ax.tick_params(axis='x',labelbottom=True)
fig.suptitle('Monthly reconstruction MAE by anonymized market',fontsize=18,fontweight='bold')
fig.supylabel('MAE (EUR/MWh) · lower is better')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)

fig.savefig(root/'results/monthly_baseline_mae_by_zone.png',dpi=180)
fig.savefig(root/'results/monthly_baseline_mae_by_zone.svg')
print('Validated 48 month/zone combinations; aggregate matches original chart.')
print('Corrected interpolation best in',sum(r[methods[2]]<min(r[m] for m in methods[:2]) for r in rows),'of 48 month/zone combinations')
