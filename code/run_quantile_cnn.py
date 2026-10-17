"""Execution-host durable Nightwatch child: success requires final independent audit."""
import os
import argparse
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
p=argparse.ArgumentParser();p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
stages=[('code/analyze_quantile_cnn.py',),('code/analyze_cnn_confidence.py',),('code/audit_quantile_cnn.py',)]
if not a.finalize_only:
    stages=[('code/quantile_cnn.py','prepare'),('code/quantile_cnn.py','smoke'),('code/quantile_cnn.py','train'),('code/quantile_cnn.py','oof')]+stages
for args in stages:
    print('STAGE',*args,flush=True)
    subprocess.run([sys.executable,*args],check=True)
