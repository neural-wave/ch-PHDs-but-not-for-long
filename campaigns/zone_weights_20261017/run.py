"""Nightwatch command: smoke, fit, audit, then archive complete evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.chdir(PROJECT)
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
for args in [('zone_blend.py', '--smoke'), ('audit.py', '--smoke'), ('zone_blend.py',), ('audit.py',)]:
    subprocess.run([sys.executable, str(HERE/args[0]), *args[1:]], check=True)
out = PROJECT/'results/zone_weights_20261017'
assert json.loads((out/'audit.json').read_text())['passed']
archive = PROJECT/'results/zone_weights_20261017_completed.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(out, arcname=out.relative_to(PROJECT))
    bundle.add(HERE, arcname=HERE.relative_to(PROJECT))
receipt = dict(archive=str(archive), bytes=archive.stat().st_size,
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), audit_passed=True)
(out/'completion.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('ZONE WEIGHTS COMPLETE', json.dumps(receipt), flush=True)
