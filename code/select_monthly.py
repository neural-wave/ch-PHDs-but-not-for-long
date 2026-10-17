"""Freeze the lowest-August-MAE complete pipeline under the requested split."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--exclude-graph',action='store_true');a=p.parse_args()
names=['monthly_base_gpu1','monthly_cat_gpu1']
if not a.exclude_graph:names.extend(['monthly_graph_gpu1','monthly_revin_gpu1','monthly_joint_blend_gpu1'])
scores={}
for name in names:
    path=Path('results',name,'metrics.json')
    if not path.exists():continue
    r=json.loads(path.read_text())
    if not r.get('offline_inference_qa','').startswith('passed'):continue
    assert r['split_counts']==dict(train=302,validation=31,test=30)
    assert r['model_fit_days']==302 and r['final_refit_deferred']
    scores[name]=r['validation']['mae']
assert scores
best=min(scores,key=scores.get)
assert Path('results',best,'model.pkl').exists(),'Selected artifact is missing: '+best
selection=dict(selected_by='august_validation_mae',best_directory=best,validation_scores=scores,
    model_fit_days=302,train_end='2026-07-31',validation_month='2026-08',test_month='2026-09')
Path('results/monthly_best.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps(selection,indent=2))
