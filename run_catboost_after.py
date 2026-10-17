"""Run the next GPU family after the tracked zone search has terminated."""
import json, os, subprocess, sys, time
from pathlib import Path

proc=Path('/proc/2122856/cmdline')
while proc.exists():
    try:cmd=proc.read_bytes().replace(b'\x00',b' ').decode()
    except FileNotFoundError:break
    if '1fedfdaa7289f909083d30a3' not in cmd:break
    print('Waiting for zone search wrapper 2122856',flush=True);time.sleep(30)
parent=json.loads(Path('results/zone_search_gpu1/metrics.json').read_text())
assert parent.get('offline_inference_qa','').startswith('passed')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    PYTHONPATH=str(Path.cwd()/'catboost_vendor'))
for smoke,name in [(True,'catboost_smoke_gpu1'),(False,'catboost_search_gpu1')]:
    cmd=[sys.executable,'catboost_campaign/train_catboost.py','--out','results/'+name]
    if smoke:cmd.append('--smoke')
    subprocess.run(cmd,env=env,check=True)
    rr=json.loads(Path('results',name,'metrics.json').read_text());assert rr.get('offline_inference_qa','').startswith('passed')
options=['search_gpu1','rich_search_gpu1','zone_search_gpu1','catboost_search_gpu1']
scores={d:json.loads(Path('results',d,'metrics.json').read_text())['validation']['mae'] for d in options}
best=min(scores,key=scores.get)
Path('results/best_search.json').write_text(json.dumps(dict(selected_by='validation_mae_only',best_directory=best,validation_scores=scores),indent=2))
print('Best validation-selected pipeline',best,'scores',scores,flush=True)
