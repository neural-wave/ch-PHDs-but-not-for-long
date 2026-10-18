"""Wait for actual remote success, create comparison, retrieve and verify backup."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'results/minimize_expanding_20261017'
HOST = 'root@93.91.156.85'
REMOTE = '/workspace/HACKATON'
RUN = '0c174584a3b9236d9f21ffd4'
ARCHIVE = 'minimize_expanding_20261017_completed.tar.gz'


def ssh(args, timeout=90):
    result = subprocess.run(['ssh', '-p', '55340', '-o', 'ConnectTimeout=15',
                             '-o', 'BatchMode=yes', HOST, shlex.join(args)],
                            check=True, text=True, capture_output=True, timeout=timeout)
    return result.stdout


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def copy(relative, destination, timeout=900):
    source = f'{HOST}:{REMOTE}/{relative}'
    subprocess.run(['scp', '-P', '55340', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes',
                    source, str(destination)], check=True, timeout=timeout)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); previous = None; failures = 0
    code = ('import json,subprocess; '
            f's=json.loads(subprocess.check_output(["/usr/bin/python3","{REMOTE}/nightwatch.py","list"])); '
            f'r=next((r for r in s["runs"] if r["id"]=="{RUN}"),None); '
            'print(json.dumps(r))')
    while True:
        assert time.monotonic()-started < 7200, 'Two-hour backup wait elapsed; verify the same remote run'
        try:
            state = json.loads(ssh(['/usr/bin/python3', '-c', code]))
            assert state is not None, 'Remote run missing; do not restart automatically'
            failures = 0
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            failures += 1
            if failures >= 6:
                raise RuntimeError('Repeated remote observation failures; training outcome unknown') from error
            print('Transient observation failure; preserving the same run', failures, flush=True)
            time.sleep(20)
            continue
        if state['status'] != previous:
            print('REMOTE SCIENTIFIC RUN', RUN, state['status'], flush=True)
            previous = state['status']
        if state['status'] == 'completed':
            assert state['exit_code'] == 0
            break
        if state['status'] in ['crashed', 'failed', 'cancelled', 'canceled']:
            raise RuntimeError(f"Remote scientific run ended {state['status']}: {state.get('crash_reason')}")
        time.sleep(20)
    (ROOT / 'remote_terminal_state.json').write_text(json.dumps(state, indent=2)+'\n')
    print('Actual remote completion confirmed; recomputing selector comparison', flush=True)
    output = ssh(['/venv/main/bin/python', REMOTE+'/campaigns/minimize_expanding_20261017/summarize_verified.py'], timeout=180)
    print(output, flush=True)
    relative = 'results/minimize_expanding_20261017/'
    metadata = ['completion.json', 'manifest.json', 'audit.json', 'comparison.json', 'REPORT.md',
                'summary.csv', 'monthly_scores.csv', 'monthly_zone_scores.csv',
                'interval_metrics.csv', 'confidence_bins.csv',
                'selector_comparison_by_training_size.csv', 'selector_interval_comparison.csv',
                'verified_selector_comparison.json', 'expanding_selector_comparison.png',
                'expanding_selector_comparison.svg']
    for name in metadata:
        copy(relative+name, ROOT/name)
    completion = json.loads((ROOT/'completion.json').read_text())
    assert completion['complete'] and completion['audit_passed']
    assert completion['archive'] == REMOTE+'/results/'+ARCHIVE
    assert re.fullmatch('[0-9a-f]{64}', completion['sha256'])
    target = PROJECT/'results'/ARCHIVE
    if not (target.exists() and target.stat().st_size == completion['bytes'] and digest(target) == completion['sha256']):
        partial = target.with_suffix(target.suffix+'.partial')
        # Preserve and reuse a full transfer after a local verifier failure.
        sha = digest(partial) if partial.exists() and partial.stat().st_size == completion['bytes'] else None
        if sha != completion['sha256']:
            copy('results/'+ARCHIVE, partial)
            sha = digest(partial)
        assert partial.stat().st_size == completion['bytes'], 'Incomplete archive transfer'
        assert sha == completion['sha256'], 'Archive checksum mismatch'
        partial.replace(target)
    receipt = dict(verified=True, scientific_run_id=RUN, archive=str(target),
                   bytes=target.stat().st_size, sha256=completion['sha256'], audit_passed=True,
                   verified_selector_comparison=json.loads((ROOT/'verified_selector_comparison.json').read_text()),
                   local_metadata_sha256={name: digest(ROOT/name) for name in metadata})
    (ROOT/'backup_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('COMPARISON AND LOCAL BACKUP VERIFIED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
