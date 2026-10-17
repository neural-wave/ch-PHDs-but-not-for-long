import os
import subprocess
import sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for script,args in [('segmented_frequency.py',['--smoke']),('segmented_frequency.py',[]),('audit_segmented_frequency.py',[])]:
    subprocess.run([sys.executable,script,*args],check=True)
