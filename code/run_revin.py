"""Durable worker command for the paired RevIN ablation."""
import os,subprocess,sys
from pathlib import Path
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH='/home/mnogales/Projects/HACKATON/catboost_vendor')
subprocess.run([sys.executable,'campaigns/revin/train_revin.py','--out','results/monthly_revin_gpu1'],env=env,check=True)
subprocess.run([sys.executable,'code/select_monthly.py'],env=env,check=True)
