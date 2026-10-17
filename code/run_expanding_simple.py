"""Execution-host tracked smoke and full expanding search."""
import os,subprocess,sys
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [['code/simple_conformal.py'],['code/expanding_simple.py','--smoke'],['code/audit_expanding_simple.py','--smoke'],['code/expanding_simple.py'],['code/audit_expanding_simple.py']]:subprocess.run([sys.executable,*args],check=True)
