"""Regenerate the retrospective rolling plan from official-data search reports."""
import json
from pathlib import Path

z=json.loads(Path('results/zone_search_gpu1/metrics.json').read_text())
c=json.loads(Path('results/catboost_search_gpu1/metrics.json').read_text())
parent=next(w for m,w in zip(c['candidates'],c['weights']) if m['name']=='parent_ensemble')
items=[]
for m,w in zip(z['candidates'],z['weights']):
    anchor=[a*b for a,b in zip(w,parent)]
    if any(anchor):items.append(dict(name=m['name'],config=m['config'],anchor=anchor))
for m,w in zip(c['candidates'],c['weights']):
    if not any(w) or m['name']=='parent_ensemble':continue
    if m['name']=='interpolation':
        item=next(v for v in items if v['name']=='interpolation');item['anchor']=[a+b for a,b in zip(item['anchor'],w)]
    else:items.append(dict(name='cat_roll_'+m['name'].split('_')[-1],config=m['config'],anchor=w))
assert all(abs(sum(x['anchor'][zz] for x in items)-1)<1e-8 for zz in range(4))
plan=dict(window_days=28,fold_train_days=[251,279,307],audit_train_days=335,penalties=[.05,.2,1.0],components=items,
    caveat='Retrospective refinement using previously selected hyperparameters; audit dates were viewed in earlier campaigns. No secret-test claim.',
    sources=['results/zone_search_gpu1/metrics.json','results/catboost_search_gpu1/metrics.json'])
Path('rolling_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Exported',len(items),'fixed component configurations and anchor weights')
