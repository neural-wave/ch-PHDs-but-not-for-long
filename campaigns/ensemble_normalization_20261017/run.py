"""Durable GPU queue, preflight, training, independent audit and archive."""
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];os.chdir(PROJECT)
sys.path.insert(0,str(PROJECT));import nightwatch
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
config=nightwatch.load_config()
while True:
    state=nightwatch.api(config,'state');previous=next((j for j in state['runs'] if j['id']=='d85b1d1b1b58a45af40c3a57'),None)
    assert previous is not None,'Cannot resolve existing GPU job; do not assume it ended'
    if previous['status'] in ['completed','failed','crashed','cancelled']:
        # Stop-tracking/cancelled does not prove GPU process termination.
        check=subprocess.run(['pgrep','-f','campaigns/linear_s4d_20261017/(train|run).py'],capture_output=True)
        if check.returncode==1:break
    print('QUEUED behind affine S4D job; actual status',previous['status'],flush=True);time.sleep(30)
print('GPU predecessor ended; starting ensemble normalization',flush=True)
for args in [('train.py','prepare'),('train.py','smoke'),('audit.py','--smoke'),('train.py','run'),('audit.py',)]:
    print('STAGE',*args,flush=True);subprocess.run([sys.executable,str(HERE/args[0]),*args[1:]],check=True)
root=PROJECT/'results/ensemble_normalization_20261017';assert json.loads((root/'audit.json').read_text())['passed']
archive=PROJECT/'results/ensemble_normalization_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(root,arcname=str(root.relative_to(PROJECT)))
    for path in json.loads((root/'manifest.json').read_text())['code_hashes']:tar.add(PROJECT/path,arcname=path)
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
(root/'completion.json').write_text(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha,audit_passed=True),indent=2))
print('ENSEMBLE NORMALIZATION COMPLETE',flush=True)
