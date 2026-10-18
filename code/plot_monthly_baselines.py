"""Plot monthly MAE from saved deterministic baseline daily errors."""
import csv,json,sys
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
methods=['hourly_mean','linear_interpolation','linear_interpolation_mean_corrected']
labels=['Repeat hourly mean','Linear interpolation','Mean-corrected interpolation']
groups=defaultdict(list)
with (root/'results/baseline_daily_metrics.csv').open() as f:
    for r in csv.DictReader(f): groups[r['delivery_date'][:7],r['method'],r['market_id']].append(float(r['mae_eur_mwh']))
months=sorted({k[0] for k in groups})
assert len(months)==12
rows=[]
for month in months:
    row={'month':month,'days':len(groups[month,methods[0],'ZONE_0'])}
    for method in methods:
        zones=[groups[month,method,f'ZONE_{i}'] for i in range(4)]
        assert all(len(z)==row['days'] for z in zones)
        row[method]=sum(sum(z)/len(z) for z in zones)/4
    rows.append(row)
assert sum(r['days'] for r in rows)==363
with (root/'results/monthly_baseline_mae.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sep=json.loads((root/'results/september_baseline_metrics.json').read_text())
for m in methods: assert abs(rows[-1][m]-sep['methods'][m]['mae_eur_mwh'])<1e-10
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,5.7),layout='constrained')
for method,label,color in zip(methods,labels,['#6b7280','#d97706','#007c91']):
    ax.plot(range(12),[r[method] for r in rows],marker='o',linewidth=2.3,label=label,color=color)
ax.set_xticks(range(12),['Oct\n2025','Nov','Dec','Jan\n2026','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'])
ax.set_ylim(bottom=0)
ax.set_ylabel('MAE (EUR/MWh) · lower is better')
ax.set_title('Monthly quarter-hour reconstruction error',loc='left',fontweight='bold',pad=18)
ax.grid(axis='y',alpha=.2)
ax.legend(loc='upper left',frameon=False)
fig.supxlabel('Equal weight across four zones; 363 days total. Same fixed baselines in every month.',fontsize=10,color='#555555')
fig.savefig(root/'results/monthly_baseline_mae.png',dpi=190)
fig.savefig(root/'results/monthly_baseline_mae.svg')
print(json.dumps(rows,indent=2))
print('Corrected interpolation best each month:',all(r[methods[2]]<min(r[m] for m in methods[:2]) for r in rows))
