"""Tracked HPO extension; reuse frozen initial winner and add 50% intervals."""
import os
import subprocess
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def run(*cmd):
    subprocess.run([sys.executable,*cmd],check=True)

run('verify_hierarchical.py')
run('add_conformal_50.py','--model-dir','results/hierarchical_vast')
run('train_hierarchical.py','--smoke','--hpo-trials','4','--out','results/hierarchical_hpo_lags_smoke_vast')
run('train_hierarchical.py','--hpo-trials','24','--parent-dir','results/hierarchical_hpo_vast','--out','results/hierarchical_hpo_lags_vast')
