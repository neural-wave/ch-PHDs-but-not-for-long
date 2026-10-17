"""Nightwatch execution-host wrapper: setup, meaningful GPU checks, smoke, full run."""
import os
import subprocess
import sys

def run(*cmd):
    subprocess.run(cmd, check=True)

os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
run('uv', 'pip', 'install', '--python', sys.executable, 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
run('uv', 'pip', 'install', '--python', sys.executable, 'numpy==2.4.6', 'pandas==2.3.3')
run(sys.executable, 'verify_hierarchical.py')
run(sys.executable, 'train_hierarchical.py', '--smoke', '--out', 'results/hierarchical_smoke_vast')
run(sys.executable, 'train_hierarchical.py', '--out', 'results/hierarchical_vast')
