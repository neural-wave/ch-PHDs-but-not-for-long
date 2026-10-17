"""Read-only cached-prediction audit: individual models vs the monthly ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--out',default='results/simplicity_audit.json')
a=p.parse_args()
rows=[]
for name in ['monthly_base_gpu1','monthly_cat_gpu1','monthly_graph_gpu1','monthly_revin_gpu1']:
    root=Path('results')/name
    report=json.loads((root/'metrics.json').read_text())
    assert report['model_fit_days']==302 and report['split_counts']==dict(train=302,validation=31,test=30)
    with np.load(root/'selection_predictions.npz',allow_pickle=True) as cache:
        truth=cache['truth'].reshape(363,4,24,4)
        preds=cache['candidates'].reshape(len(report['candidates']),363,4,24,4)
        for meta,pred in zip(report['candidates'],preds):
            model=meta['name']
            if model in ['parent_ensemble','interpolation','flat']:
                continue
            family=next((f for f in ['ridge','trees','boost','net','cat','graph','revin'] if model.startswith(f+'_')),None)
            if family is None:
                continue
            rows.append(dict(name=model,family=family,origin=name,config=meta['config'],
                selection15_mae=float(np.abs(pred[302:317]-truth[302:317]).mean()),
                august_mae=float(np.abs(pred[302:333]-truth[302:333]).mean()),
                september_mae=float(np.abs(pred[333:]-truth[333:]).mean())))
families={f:min([r for r in rows if r['family']==f],key=lambda r:r['august_mae']) for f in sorted({r['family'] for r in rows})}
single=min(rows,key=lambda r:r['august_mae'])
ensemble=json.loads(Path('results/monthly_joint_blend_gpu1/metrics.json').read_text())
result=dict(selection='all August MAE only; no September selection',best_by_family=families,best_single=single,
    ensemble=dict(august_mae=ensemble['validation']['mae'],september_mae=ensemble['test']['mae']),candidates=rows)
for r in list(families.values())+[single]:
    r['ensemble_test_gain_absolute']=r['september_mae']-result['ensemble']['september_mae']
    r['ensemble_test_gain_percent']=100*r['ensemble_test_gain_absolute']/r['september_mae']
Path(a.out).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
