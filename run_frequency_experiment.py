"""Remote Nightwatch command: preflight, smoke, full study, independent audit."""
import os
import subprocess
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for script,args in [('frequency_experiment.py',['--smoke']),('frequency_experiment.py',[]),('audit_frequency_experiment.py',[])]:
    subprocess.run([sys.executable,script,*args],check=True)
