"""Durable sequential follow-up; PID identity checked to avoid unrelated PID reuse."""
import json, os, subprocess, sys, time
from pathlib import Path

def main():
    pid=2112796
    proc=Path('/proc')/str(pid)/'cmdline'
    while proc.exists():
        try: cmd=proc.read_bytes().replace(b'\x00',b' ').decode()
        except FileNotFoundError: break
        if 'train_search.py' not in cmd: break
        print('Waiting for initial search PID',pid,flush=True)
        time.sleep(30)
    first=Path('results/search_gpu1/metrics.json')
    assert first.exists(), 'Initial search did not produce metrics; inspect its Nightwatch result'
    data=json.loads(first.read_text())
    assert data.get('offline_inference_qa','').startswith('passed'), 'Initial search incomplete or QA failed'
    # Keep this printed summary to validation only; do not tune from holdout scores.
    print('Initial search validation',data['validation'],flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    subprocess.run([sys.executable,'rich_campaign/train_search.py','--rich','--smoke','--out','results/rich_smoke_gpu1'],env=env,check=True)
    subprocess.run([sys.executable,'rich_campaign/train_search.py','--rich','--out','results/rich_search_gpu1'],env=env,check=True)
    rich=json.loads(Path('results/rich_search_gpu1/metrics.json').read_text())
    assert rich.get('offline_inference_qa','').startswith('passed')
    # Record best using validation only. Holdout remains a diagnostic, not selection input.
    best='rich_search_gpu1' if rich['validation']['mae']<data['validation']['mae'] else 'search_gpu1'
    Path('results/best_search.json').write_text(json.dumps({'selected_by':'validation_mae_only',
        'best_directory':best,'initial_validation_mae':data['validation']['mae'],
        'rich_validation_mae':rich['validation']['mae']},indent=2))
    print('Best validation-selected pipeline:',best,flush=True)

if __name__=='__main__': main()
