"""Run a tiny six-network fit and offline inference from a standalone copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output}: use a new output directory')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='neural-wave-jury-') as temporary:
        temporary = Path(temporary)
        package = temporary / 'standalone_package'
        ignored = shutil.ignore_patterns('.venv', '__pycache__', '*.pyc', '*.pt', '*.npz',
                                         'model_checkpoints', 'verification', 'wheelhouse',
                                         'installed-environment.txt')
        shutil.copytree(ROOT, package, ignore=ignored)
        work = temporary / 'unrelated_working_directory'
        work.mkdir()
        artifacts = temporary / 'trial_artifacts'
        artifacts.mkdir()
        logs = artifacts / 'logs'
        logs.mkdir()
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        calls = []

        def run(label, script, *arguments, failure_expected=False):
            command = [sys.executable, '-I', str(package / 'src/offline_entrypoint.py'),
                       str(package / 'src' / script), *map(str, arguments)]
            completed = subprocess.run(command, cwd=work, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            (logs / (label + '.log')).write_text(completed.stdout)
            calls.append(dict(check=label, script=script, returncode=completed.returncode,
                              expected_failure=failure_expected))
            if (completed.returncode != 0) != failure_expected:
                output.mkdir(parents=True, exist_ok=True)
                shutil.copytree(logs, output / 'logs', dirs_exist_ok=True)
                raise RuntimeError(f'{label} failed; see {output / "logs" / (label + ".log")}\n' + completed.stdout[-4000:])
            print('PASS', label, flush=True)

        run('train_six_components', 'train_ensemble.py', '--mode', 'trial', '--device', 'cpu',
            '--output', artifacts)
        run('replay_causality_and_isolation', 'verify_invariants.py', '--checkpoint-dir', artifacts)
        fixture = artifacts / 'fixture'
        model = artifacts / 'model.pt'
        checkpoint_hash = sha256(model)

        def predict(label, hourly, history, *, allow_trial=True, failure_expected=False, quantiles=True):
            point = fixture / (label + '_predictions.csv')
            bands = fixture / (label + '_quantiles.csv')
            arguments = ['--model', model, '--hourly', fixture / hourly,
                         '--history', fixture / history, '--output', point]
            if allow_trial:
                arguments.append('--allow-trial')
            if quantiles:
                arguments.extend(['--quantiles-output', bands])
            run(label, 'predict_prices.py', *arguments, failure_expected=failure_expected)
            if failure_expected:
                if point.exists() or bands.exists():
                    raise RuntimeError('Rejected input produced an output CSV')
            else:
                arguments = ['--hourly', fixture / hourly, '--predictions', point,
                             '--report', artifacts / (label + '_contract.json')]
                if quantiles:
                    arguments.extend(['--quantiles', bands])
                run(label + '_contract', 'check_predictions.py', *arguments)

        predict('single_day', 'test_hourly.csv', 'history_15min.csv')
        predict('multiple_days', 'test_hourly_multiple.csv', 'history_15min.csv')
        predict('empty_history', 'test_hourly.csv', 'empty_history.csv', quantiles=False)
        predict('requires_explicit_trial_flag', 'test_hourly.csv', 'history_15min.csv',
                allow_trial=False, failure_expected=True)
        for label, hourly, history in [
            ('reject_duplicate_hour', 'duplicate_hourly.csv', 'history_15min.csv'),
            ('reject_missing_hour', 'incomplete_hourly.csv', 'history_15min.csv'),
            ('reject_nonfinite_hour', 'nonfinite_hourly.csv', 'history_15min.csv'),
            ('reject_nonprior_history', 'test_hourly.csv', 'nonprior_history.csv'),
            ('reject_duplicate_history', 'test_hourly.csv', 'duplicate_history.csv'),
        ]:
            predict(label, hourly, history, failure_expected=True)
        if sha256(model) != checkpoint_hash:
            raise RuntimeError('Inference changed the checkpoint')
        training = json.loads((artifacts / 'training_report.json').read_text())
        contracts = {name: json.loads((artifacts / (name + '_contract.json')).read_text())
                     for name in ['single_day', 'multiple_days', 'empty_history']}
        report = dict(passed=True, artifact_kind='isolated_trial',
                      scope='Execution and data-contract check only; no final accuracy claim or full-budget retrain.',
                      standalone_copy_outside_repository=True, unrelated_working_directory=True,
                      isolated_python_flag=True, socket_networking_disabled=True,
                      checkpoint_unchanged_by_inference=True, model_sha256=checkpoint_hash,
                      training_days=4, calibration_days=2, components=6, epochs_per_component=1,
                      environment=training['environment'], checks=calls, csv_contracts=contracts,
                      seconds=time.perf_counter() - started)
        (artifacts / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (artifacts / 'README.txt').write_text(
            'ISOLATED EXECUTION TRIAL ONLY. Four fit dates, two calibration dates, one epoch per component.\n'
            'These weights are not the original frozen champion. Use --allow-trial for inference.\n'
            'See verification.json, training_report.json, invariants.json and logs/ for evidence.\n')
        shutil.copytree(artifacts, output, dirs_exist_ok=True)
    print('ISOLATED TRIAL PASSED:', output / 'verification.json', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'model_checkpoints/jury_trial')
    args = parser.parse_args()
    verify(args.output)
