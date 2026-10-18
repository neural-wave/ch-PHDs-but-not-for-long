"""Queue joint-node model experiments behind the existing monthly GPU job."""
import json,os,subprocess,sys,time
from pathlib import Path

proc=Path('/proc/2147568/cmdline')
while proc.exists():
    try:cmd=proc.read_bytes().replace(b'\x00',b' ').decode()
    except FileNotFoundError:break
    if '33ca46c2ab0f4ebb14e99ea2' not in cmd:break
    print('Waiting for monthly wrapper 2147568',flush=True);time.sleep(30)
assert Path('results/monthly_best.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    PYTHONPATH='/home/mnogales/Projects/HACKATON/catboost_vendor')
for smoke,name in [(True,'monthly_graph_smoke_gpu1'),(False,'monthly_graph_gpu1')]:
    cmd=[sys.executable,'campaigns/graph/train_graph.py','--out','results/'+name]
    if smoke:cmd.append('--smoke')
    subprocess.run(cmd,env=env,check=True)
    rr=json.loads(Path('results',name,'metrics.json').read_text());assert rr.get('offline_inference_qa','').startswith('passed')
names=['monthly_base_gpu1','monthly_cat_gpu1','monthly_graph_gpu1']
scores={d:json.loads(Path('results',d,'metrics.json').read_text())['validation']['mae'] for d in names}
best=min(scores,key=scores.get)
selection=dict(selected_by='august_validation_mae',best_directory=best,validation_scores=scores,model_fit_days=302,
    train_end='2026-07-31',validation_month='2026-08',test_month='2026-09')
Path('results/monthly_best.json').write_text(json.dumps(selection,indent=2)+'\n')
print('MONTHLY WINNER by AUGUST only',selection,flush=True)
