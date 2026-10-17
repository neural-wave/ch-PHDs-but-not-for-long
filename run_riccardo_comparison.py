"""Execution-host Nightwatch child: smoke, full run, audit, archive."""
import os
import subprocess
import sys
import tarfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
root=Path(__file__).resolve().parent
os.chdir(root)
for args in [['compare_riccardo.py','--smoke'],['audit_riccardo_comparison.py','--smoke'],
             ['compare_riccardo.py'],['audit_riccardo_comparison.py']]:
    subprocess.run([sys.executable,*args],check=True)
with tarfile.open(root/'results/riccardo_comparison_20261017_completed.tar.gz','w:gz') as archive:
    for path in ['results/riccardo_comparison_20261017','results/riccardo_comparison_20261017_smoke',
                 'campaigns/riccardo_comparison_20261017','compare_riccardo.py',
                 'audit_riccardo_comparison.py','run_riccardo_comparison.py']:
        archive.add(root/path,arcname=path)
print('COMPLETED: both models, eight folds, independent audits and archive',flush=True)
