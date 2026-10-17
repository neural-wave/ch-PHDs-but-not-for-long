"""Smoke, full zone-specific search, and validation-only pipeline selection."""
import json, os, subprocess, sys
from pathlib import Path

env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for smoke,name in [(True,'zone_smoke_gpu1'),(False,'zone_search_gpu1')]:
    cmd=[sys.executable,'zonal_campaign/train_search.py','--rich','--zone-ensemble','--out','results/'+name]
    if smoke: cmd.append('--smoke')
    subprocess.run(cmd,env=env,check=True)
    report=json.loads(Path('results',name,'metrics.json').read_text())
    assert report.get('offline_inference_qa','').startswith('passed')
options=['search_gpu1','rich_search_gpu1','zone_search_gpu1']
scores={d:json.loads(Path('results',d,'metrics.json').read_text())['validation']['mae'] for d in options}
best=min(scores,key=scores.get)
Path('results/best_search.json').write_text(json.dumps(dict(selected_by='validation_mae_only',best_directory=best,validation_scores=scores),indent=2))
print('Best validation-selected pipeline',best,'scores',scores,flush=True)
