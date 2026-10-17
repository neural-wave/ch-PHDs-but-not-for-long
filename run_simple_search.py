"""Tracked remote preflight, concurrency benchmark, then full search."""
import json,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def run(*args):subprocess.run(args,check=True)
run(sys.executable,'-c','import torch; assert torch.cuda.is_available(); x=torch.randn(128,128,device="cuda"); assert torch.isfinite(x@x).all(); print(torch.__version__,torch.cuda.get_device_name())')
run(sys.executable,'simple_search.py','--prepare')
run(sys.executable,'simple_search.py','--smoke')
profile=json.loads(Path('results/simple_5090/profile.json').read_text())
workers=min(profile,key=lambda p:p['seconds'])['workers']
Path('results/simple_5090/concurrency.json').write_text(json.dumps(dict(workers=workers,criterion='lowest wall time on identical 16-job five-epoch workload',profile=profile),indent=2))
run(sys.executable,'simple_search.py','--workers',str(workers))
assert Path('results/simple_5090/metrics.json').exists()
