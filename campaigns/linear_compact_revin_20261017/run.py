"""Durable stages; success requires verification, requested plots and archive."""
import os,sys,subprocess,tarfile,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];os.chdir(PROJECT)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
# A tracked dependency wait avoids concurrent changes and duplicated preparation.
import time
parent=PROJECT/'results/champion_revin_20261017'
start=time.monotonic()
while not (parent/'completion.json').exists():
    if time.monotonic()-start>3600:raise RuntimeError('Champion RevIN dependency did not complete within1hour')
    try:os.kill(15184,0)
    except ProcessLookupError:raise RuntimeError('Champion RevIN worker ended without success receipt')
    time.sleep(10)
for args in [('audit.py','--edge-only'),('train.py','prepare'),('train.py','smoke'),('audit.py','--smoke'),('train.py','run'),('audit.py',),('report.py',)]:
    print('STAGE',*args,flush=True);subprocess.run([sys.executable,str(PROJECT/'run.py'),str(HERE/args[0]),*args[1:]],check=True)
root=PROJECT/'results/linear_compact_revin_20261017';assert json.loads((root/'audit.json').read_text())['passed'];archive=PROJECT/'results/linear_compact_revin_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(root,arcname=str(root.relative_to(PROJECT)))
    for path in json.loads((root/'manifest.json').read_text())['code_hashes']:tar.add(PROJECT/path,arcname=path)
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
(root/'completion.json').write_text(json.dumps(dict(passed=True,archive=str(archive),sha256=sha,bytes=archive.stat().st_size),indent=2));print('LINEAR COMPACT REVIN COMPLETE',flush=True)
