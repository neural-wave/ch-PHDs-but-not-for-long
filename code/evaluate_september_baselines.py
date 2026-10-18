"""Aggregate unchanged deterministic baseline predictions on September 2026."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parents[1]
methods=['hourly_mean','linear_interpolation','linear_interpolation_mean_corrected']
errors=defaultdict(list)
keys=set()
with (root/'results/baseline_predictions.csv').open() as f:
    for r in csv.DictReader(f):
        if not r['delivery_date'].startswith('2026-09-'): continue
        key=(r['delivery_date'],r['market_id'],int(r['hour_index']),int(r['quarter_index']))
        assert key not in keys
        keys.add(key)
        y=float(r['observed_price_eur_mwh'])
        for m in methods: errors[m,r['market_id']].append(float(r[m])-y)
assert len(keys)==30*4*96
assert len({k[0] for k in keys})==30
summary={'period':'2026-09-01 through 2026-09-30','days':30,'rows':len(keys),'methods':{}}
for m in methods:
    zones={z:sum(abs(v) for v in errors[m,z])/len(errors[m,z]) for z in ['ZONE_0','ZONE_1','ZONE_2','ZONE_3']}
    assert all(len(errors[m,z])==2880 for z in zones)
    summary['methods'][m]={'mae_eur_mwh':sum(zones.values())/4,'mae_by_zone':zones,'rmse_eur_mwh':math.sqrt(sum(v*v for z in zones for v in errors[m,z])/len(keys))}
# Independently cross-check saved daily MAEs.
with (root/'results/baseline_daily_metrics.csv').open() as f:
    daily=list(csv.DictReader(f))
for m in methods:
    selected=[float(r['mae_eur_mwh']) for r in daily if r['method']==m and r['delivery_date'].startswith('2026-09-')]
    assert len(selected)==120
    assert abs(sum(selected)/120-summary['methods'][m]['mae_eur_mwh'])<1e-10
summary['corrected_mae_reduction_percent']=100*(1-summary['methods'][methods[2]]['mae_eur_mwh']/summary['methods'][methods[0]]['mae_eur_mwh'])
(root/'results/september_baseline_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
