"""Durable local archive download and checksum finalization; no model fitting."""
import hashlib,json,subprocess
from pathlib import Path
project=Path(__file__).resolve().parents[2]
root=project/'results/ensemble_normalization_20261017'
receipt=json.loads((root/'completion.json').read_text())
archive=project/'results/ensemble_normalization_20261017_completed.tar.gz'
subprocess.run(['scp','-P','55340','root@93.91.156.85:'+receipt['archive'],str(archive)],check=True)
assert archive.stat().st_size==receipt['bytes'],'Incomplete archive'
sha=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
assert sha.hexdigest()==receipt['sha256'],'Archive checksum mismatch'
verified=dict(verified=True,bytes=archive.stat().st_size,sha256=sha.hexdigest(),archive=str(archive),audit_passed=True)
(root/'backup_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
print(json.dumps(verified),flush=True)
