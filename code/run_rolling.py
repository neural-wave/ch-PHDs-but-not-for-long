"""Smoke followed by full rolling campaign on physical Ando GPU1."""
import os,subprocess,sys

env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    PYTHONPATH='/home/mnogales/Projects/HACKATON/catboost_vendor')
for smoke,name in [(True,'rolling_smoke_gpu1'),(False,'rolling_search_gpu1')]:
    cmd=[sys.executable,'code/train_rolling.py','--out','results/'+name]
    if smoke:cmd.append('--smoke')
    subprocess.run(cmd,env=env,check=True)
