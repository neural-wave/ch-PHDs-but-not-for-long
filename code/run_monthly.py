"""Requested calendar-month validation/test, preserving train-only fitted models."""
import json,os,subprocess,sys
from pathlib import Path

env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    PYTHONPATH='/home/mnogales/Projects/HACKATON/catboost_vendor')
stages=[('code/train_search.py','monthly_smoke_gpu1',True),('code/train_search.py','monthly_base_gpu1',False),
        ('code/train_catboost.py','monthly_cat_smoke_gpu1',True),('code/train_catboost.py','monthly_cat_gpu1',False)]
for script,name,smoke in stages:
    cmd=[sys.executable,'campaigns/monthly/'+Path(script).name,'--defer-refit','--out','results/'+name]
    if script=='code/train_search.py':cmd+=['--rich','--zone-ensemble','--calendar-split']
    else:cmd+=['--parent-dir','results/monthly_base_gpu1']
    if smoke:cmd.append('--smoke')
    subprocess.run(cmd,env=env,check=True)
    r=json.loads(Path('results',name,'metrics.json').read_text())
    assert r['split_counts']==dict(train=302,validation=31,test=30)
    assert r['model_fit_days']==302 and r['final_refit_deferred']
    assert r.get('offline_inference_qa','').startswith('passed')
names=['monthly_base_gpu1','monthly_cat_gpu1']
scores={d:json.loads(Path('results',d,'metrics.json').read_text())['validation']['mae'] for d in names}
best=min(scores,key=scores.get)
selection=dict(selected_by='august_validation_mae',best_directory=best,validation_scores=scores,
    train_end='2026-07-31',validation_month='2026-08',test_month='2026-09',model_fit_days=302)
Path('results/monthly_best.json').write_text(json.dumps(selection,indent=2)+'\n')
print('MONTHLY WINNER selected by AUGUST only:',selection,flush=True)
