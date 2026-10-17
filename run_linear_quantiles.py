"""Execution-host Nightwatch worker for the offline simple quantile experiment."""
import os,subprocess,sys
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def run(*args):subprocess.run([sys.executable,*args],check=True)
run('verify_linear_quantiles.py')
run('train_linear_quantiles.py','--smoke','--out','results/linear_quantiles_smoke_vast')
run('train_linear_quantiles.py','--out','results/linear_quantiles_vast')
