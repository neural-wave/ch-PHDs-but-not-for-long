"""Durable tracked smoke/full expanding-window runner."""
import os,subprocess,sys
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def run(*args):subprocess.run(args,check=True)
run('uv','pip','install','--python',sys.executable,'matplotlib')
run(sys.executable,'verify_hierarchical.py')
run(sys.executable,'train_expanding.py','--smoke','--out','results/expanding_smoke_vast')
run(sys.executable,'train_expanding.py','--out','results/expanding_vast')
