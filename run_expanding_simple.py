"""Execution-host tracked smoke and full expanding search."""
import os,subprocess,sys
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [['simple_conformal.py'],['expanding_simple.py','--smoke'],['audit_expanding_simple.py','--smoke'],['expanding_simple.py'],['audit_expanding_simple.py']]:subprocess.run([sys.executable,*args],check=True)
