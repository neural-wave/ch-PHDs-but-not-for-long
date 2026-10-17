"""Remote detached execution; completion requires audited metrics and archive."""
import hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];os.chdir(PROJECT)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
subprocess.run([sys.executable,str(HERE/'train.py'),'prepare'],check=True)
root=PROJECT/'results/normalization_20261017'
stages=[] if (root/'smoke/audit.json').exists() and json.loads((root/'smoke/audit.json').read_text())['passed'] else [('train.py','smoke'),('audit.py','--smoke')]
for args in stages+[('train.py','run'),('audit.py',)]:
    print('STAGE',*args,flush=True);subprocess.run([sys.executable,str(HERE/args[0]),*args[1:]],check=True)
root=PROJECT/'results/normalization_20261017';assert json.loads((root/'audit.json').read_text())['passed']
archive=PROJECT/'results/normalization_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(root,arcname=str(root.relative_to(PROJECT)))
    manifest=json.loads((root/'manifest.json').read_text())
    for path in manifest['code_hashes']:tar.add(PROJECT/path,arcname=path)
    tar.add(root/'manifest.json',arcname='normalization_source_manifest.json')
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
(root/'completion.json').write_text(json.dumps(dict(archive=str(archive),sha256=sha,bytes=archive.stat().st_size,audit_passed=True),indent=2))
print('NORMALIZATION COMPLETE',flush=True)
