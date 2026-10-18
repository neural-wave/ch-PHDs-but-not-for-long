"""Tracked replacement campaign; final success requires audit and recovery archive."""
import os,sys,subprocess,tarfile,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];os.chdir(PROJECT)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
for args in [('train.py','prepare'),('train.py','smoke'),('audit.py','--smoke'),('train.py','run'),('audit.py',)]:
    print('STAGE',*args,flush=True);subprocess.run([sys.executable,str(HERE/args[0]),*args[1:]],check=True)
root=PROJECT/'results/ssm_monthly_20261017';assert json.loads((root/'audit.json').read_text())['passed']
archive=PROJECT/'results/ssm_monthly_20261017_completed.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(root,arcname=root.relative_to(PROJECT));tar.add(HERE,arcname=HERE.relative_to(PROJECT))
    for name in ['ssm_models.py','ssm_experiment.py','VALIDATION_PROTOCOL.md']:tar.add(PROJECT/name,arcname=name)
(root/'completion.json').write_text(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,audit_passed=True),indent=2))
print('SSM MONTHLY COMPLETE',flush=True)
