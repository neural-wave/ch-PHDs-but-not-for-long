"""Detached execution with quantile audit and checkpoint recovery archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];os.chdir(PROJECT)
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('trainq.py','prepare'),('trainq.py','smoke'),('auditq.py','--smoke'),('trainq.py','run'),('auditq.py',)]:
    subprocess.run([sys.executable,str(HERE/args[0]),*args[1:]],check=True)
out=PROJECT/'results/monthly_quantiles_20261017';assert json.loads((out/'audit.json').read_text())['passed']
archive=PROJECT/'results/monthly_quantiles_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as bundle:
    bundle.add(out,arcname=out.relative_to(PROJECT));bundle.add(HERE,arcname=HERE.relative_to(PROJECT))
    bundle.add(PROJECT/'campaigns/monthly_ratio_20261017',arcname='campaigns/monthly_ratio_20261017')
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),audit_passed=True)
(out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n');print('QUANTILE COMPLETE',json.dumps(receipt),flush=True)
