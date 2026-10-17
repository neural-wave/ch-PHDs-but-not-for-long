"""Execution-host durable Nightwatch child: success requires final independent audit."""
import os
import argparse
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
p=argparse.ArgumentParser();p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
stages=[('analyze_quantile_cnn.py',),('analyze_cnn_confidence.py',),('audit_quantile_cnn.py',)]
if not a.finalize_only:
    stages=[('quantile_cnn.py','prepare'),('quantile_cnn.py','smoke'),('quantile_cnn.py','train'),('quantile_cnn.py','oof')]+stages
for args in stages:
    print('STAGE',*args,flush=True)
    subprocess.run([sys.executable,*args],check=True)
