"""Durable remote run; success requires full audit and recovery archive."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.chdir(PROJECT)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
root = PROJECT / 'results/minimize_expanding_20261017'
root.mkdir(parents=True, exist_ok=True)
# This PID is verified at launch; do not contend with the earlier GPU campaign.
dependency = int(sys.argv[1]) if len(sys.argv) > 1 else 0
started = time.monotonic()
while dependency and Path(f'/proc/{dependency}').exists():
    assert time.monotonic()-started < 1800, 'Dependency still running after bounded 30-minute wait'
    if 'campaigns/linear_s4d_20261017/run.py' not in Path(f'/proc/{dependency}/cmdline').read_bytes().decode().replace('\0', ' '):
        raise RuntimeError('Dependency PID reused; verify execution state')
    print('WAITING for earlier S4D campaign', dependency, flush=True)
    time.sleep(15)
for args in [('train.py', 'prepare'), ('train.py', 'smoke'), ('audit.py', '--smoke'),
             ('train.py', 'run'), ('audit.py',)]:
    print('STAGE', *args, flush=True)
    subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]], check=True)
assert json.loads((root / 'audit.json').read_text())['passed']
archive = PROJECT / 'results/minimize_expanding_20261017_completed.tar.gz'
manifest = json.loads((root / 'manifest.json').read_text())
sources = root / 'execution_sources'
for name in manifest['code_hashes']:
    dest = sources / name; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((PROJECT / name).read_bytes())
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(root, arcname=str(root.relative_to(PROJECT)))
    bundle.add(HERE, arcname=str(HERE.relative_to(PROJECT)))
    old = PROJECT / 'results/monthly_quantiles_20261017'
    for filename in ['manifest.json', 'audit.json', 'summary.csv']:
        bundle.add(old / filename, arcname=str((old / filename).relative_to(PROJECT)))
    # Include the frozen controls' replay dependencies in the recovery archive.
    for folder in sorted((old / 'folds').iterdir()):
        for filename in ['data.npz', 'manifest.json', 'selection.json']:
            bundle.add(folder / filename, arcname=str((folder / filename).relative_to(PROJECT)))
        selection = json.loads((folder / 'selection.json').read_text())['by_recipe']
        ids = {str(r['config']['id']) for r in selection.values()
               if isinstance(r['config']['id'], int)}
        ids.update(['blend_uniform', 'blend_ratio'])
        for cid in ids:
            bundle.add(folder / 'trials' / cid, arcname=str((folder / 'trials' / cid).relative_to(PROJECT)))
with archive.open('rb') as stream:
    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = dict(archive=str(archive), sha256=sha, bytes=archive.stat().st_size,
               audit_passed=True, complete=True)
(root / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('EXPANDING OPTIMIZATION COMPLETE', json.dumps(receipt), flush=True)
