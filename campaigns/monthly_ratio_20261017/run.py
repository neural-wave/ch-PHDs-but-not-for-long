"""Nightwatch-managed execution host wrapper, including audit and recovery archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
os.chdir(PROJECT)
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for command in [
    [str(HERE/'experiment.py'),'prepare'],
    [str(HERE/'experiment.py'),'smoke'],
    [str(HERE/'audit.py'),'--smoke'],
    [str(HERE/'experiment.py'),'run'],
    [str(HERE/'audit.py')],
]:
    subprocess.run([sys.executable,*command],check=True)
output=PROJECT/'results/monthly_ratio_20261017'
assert json.loads((output/'audit.json').read_text())['passed']
archive=PROJECT/'results/monthly_ratio_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as bundle:
    bundle.add(output,arcname=output.relative_to(PROJECT))
    bundle.add(HERE,arcname=HERE.relative_to(PROJECT))
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,
             sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),audit_passed=True)
(output/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('COMPLETE',json.dumps(receipt),flush=True)
