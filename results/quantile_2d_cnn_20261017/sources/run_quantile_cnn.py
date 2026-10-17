"""Execution-host durable Nightwatch child: success requires final independent audit."""
import os
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('quantile_cnn.py','prepare'),('quantile_cnn.py','smoke'),('quantile_cnn.py','train'),
             ('quantile_cnn.py','oof'),('analyze_quantile_cnn.py',),('audit_quantile_cnn.py',)]:
    print('STAGE',*args,flush=True)
    subprocess.run([sys.executable,*args],check=True)
