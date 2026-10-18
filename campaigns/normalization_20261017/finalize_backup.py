"""Finalize the existing local transfer, with actual SHA256 success/failure."""
import hashlib,json,os,sys,time
from pathlib import Path

project=Path(__file__).resolve().parents[2]
root=project/'results/normalization_20261017'
receipt=json.loads((root/'completion.json').read_text())
archive=project/'results/normalization_20261017_completed.tar.gz'
transfer_pid=int(sys.argv[1])
while True:
    try:os.kill(transfer_pid,0)
    except ProcessLookupError:break
    time.sleep(2)
assert archive.stat().st_size==receipt['bytes'],'Transfer ended with an incomplete archive'
sha=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
assert sha.hexdigest()==receipt['sha256'],'Archive checksum mismatch'
verified=dict(verified=True,bytes=archive.stat().st_size,sha256=sha.hexdigest(),archive=str(archive),audit_passed=True)
(root/'backup_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
print(json.dumps(verified),flush=True)
