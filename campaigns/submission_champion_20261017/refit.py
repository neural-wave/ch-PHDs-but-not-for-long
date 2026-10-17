"""Fresh fixed-budget champion fit, independent held-out calibration and export."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'campaigns/monthly_quantiles_20261017'))
import trainq as q
import numpy as np
import pandas as pd
import torch

ROOT = PROJECT / 'results/submission_champion_20261017'
STAGE = ROOT / 'bundle'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run():
    q.p.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'completion.json').exists(), 'Completed output exists; do not overwrite'
    recipe = json.loads((HERE / 'recipe.json').read_text())
    source = PROJECT / 'train_prices_15min.csv'
    assert sha(source) == recipe['official_sha256'] == q.p.SHA
    for path, digest in recipe['source_hashes'].items():
        assert sha(PROJECT / path) == digest, path
    raw_path = PROJECT / 'results/monthly_quantiles_20261017/raw.npz'
    old_manifest = json.loads((PROJECT / 'results/monthly_quantiles_20261017/manifest.json').read_text())
    assert sha(raw_path) == old_manifest['feature_source_sha256']
    raw = dict(np.load(raw_path))
    dates = raw['dates']; mask = dates < recipe['calibration_start']
    assert mask.sum() == 348 and (~mask).sum() == 15
    assert dates[0] == '2025-10-01' and dates[-1] == '2026-09-30'
    # Verify cache causality at the fitting/calibration boundary and last day.
    for index in [0, 347, 348, 362]:
        changed = raw['y'].copy(); changed[index:] += np.arange(4) * 123456
        probe = q.p.features(dates[index:index+1].astype(object), raw['h'][index:index+1],
                             dates.astype(object), changed)
        trunc = q.p.features(dates[index:index+1].astype(object), raw['h'][index:index+1],
                             dates[:index].astype(object), raw['y'][:index])
        for key in probe:
            np.testing.assert_allclose(probe[key], raw[key][index:index+1], atol=1e-9, rtol=0)
            np.testing.assert_allclose(trunc[key], raw[key][index:index+1], atol=1e-9, rtol=0)
    data = {key: np.clip(raw[key], -15, 15).astype('float32')
            for key in ['rich', 'compact', 'cnn', 'curve']}
    data['compact'] = np.concatenate([data['compact'], np.broadcast_to(
        np.eye(4)[None, :, None, :], (len(dates), 4, 24, 4))], -1).astype('float32')
    for key in ['rich', 'compact']:
        data[key + '_knots'] = np.quantile(data[key][mask].reshape(-1, data[key].shape[-1]),
                                          [.25, .5, .75], axis=0).astype('float32')
    weights, details = q.p.weights_for(dates, raw['y'], raw['b'], mask, 'ratio')
    assert len(weights) == 348
    models = []; cal_q = np.zeros((*raw['y'][~mask].shape, 11), dtype='float64')
    target = torch.tensor((raw['y'][mask]-raw['b'][mask])/20, dtype=torch.float32, device='cuda')
    tau = torch.tensor(q.TAUS, dtype=torch.float32, device='cuda')
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device='cuda')
    for leaf in recipe['leaves']:
        cfg = leaf['config']; q.p.setup(cfg['seed'])
        net = q.Quantiles(cfg, data)
        net.initialize((raw['y'][mask]-raw['b'][mask])/20); net.cuda()
        x = torch.tensor(data[net.feature][mask], device='cuda')
        curve = torch.tensor(data['curve'][mask], device='cuda')
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=.01)
        for epoch in range(1, leaf['best_epoch']+1):
            net.train()
            for ids in torch.randperm(348, device='cuda').split(32):
                predicted = net(x[ids], curve[ids]); error = target[ids, ..., None] - predicted
                mae = error[..., 5].abs().mean((1, 2, 3))
                pinball = torch.maximum(tau*error, (tau-1)*error).mean((1, 2, 3, 4))
                loss = ((mae + cfg['quantile_weight']*pinball)*weight_tensor[ids]).mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
            if epoch % 20 == 0:
                print('FIT', cfg['family'], cfg['seed'], epoch, '/', leaf['best_epoch'], flush=True)
        net.eval()
        with torch.no_grad():
            out = net(torch.tensor(data[net.feature][~mask], device='cuda'),
                      torch.tensor(data['curve'][~mask], device='cuda')).cpu().numpy()
        prediction = q.physical(out, raw['b'][~mask])
        cal_q += leaf['weight'] * prediction
        models.append(dict(config=cfg, weight=leaf['weight'], epochs=leaf['best_epoch'],
                           state={key: value.cpu().clone() for key, value in net.state_dict().items()}))
        print('COMPONENT COMPLETE', cfg['family'], cfg['seed'], flush=True)
    bands = q.intervals(cal_q, raw['y'][~mask], cal_q)
    calibration = {str(level): {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in item.items()
                              if key not in ['lower', 'upper', 'cal_lower', 'cal_upper']}
                   for level, item in bands.items()}
    STAGE.mkdir(parents=True, exist_ok=True)
    source_dest = STAGE / 'sources'; source_dest.mkdir(exist_ok=True)
    for filename in recipe['runtime_sources']:
        shutil.copy2(PROJECT / filename, source_dest / Path(filename).name)
    shutil.copy2(HERE / 'runtime.py', STAGE / 'run.py')
    shutil.copy2(HERE / 'SPECIFICATION_CARD.md', STAGE / 'SPECIFICATION_CARD.md')
    shutil.copy2(HERE / 'README.md', STAGE / 'README.md')
    shutil.copy2(HERE / 'requirements.txt', STAGE / 'requirements.txt')
    bundle = dict(schema_version=1, recipe=recipe, models=models, calibration=calibration,
                  training_days=348, calibration_days=15, training_month_baselines=details)
    torch.save(bundle, STAGE / 'model.pt')
    q.p.write(STAGE / 'calibration.json', calibration)
    q.p.write(STAGE / 'recipe.json', recipe)
    np.savez(ROOT / 'calibration_predictions.npz', dates=dates[~mask], observed=raw['y'][~mask], quantiles=cal_q)
    # Isolated runtime imports; packaged sources must work without repository modules.
    verify_code = '''import importlib.util,sys,numpy as np,torch
from pathlib import Path
stage=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('champion_runtime',stage/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
b=torch.load(stage/'model.pt',map_location='cpu',weights_only=False)
raw=np.load(sys.argv[2]);expected=np.load(sys.argv[3]);dates=expected['dates']
pred=r.predict_arrays(b,dates.astype(object),raw['h'][-15:],raw['dates'].astype(object),raw['y'])
delta=float(np.abs(pred-expected['quantiles']).max());assert delta<.0001,delta
print('ISOLATED_REPLAY',delta)
'''
    (ROOT / 'verify_runtime.py').write_text(verify_code)
    subprocess.run([sys.executable, '-I', str(ROOT/'verify_runtime.py'), str(STAGE),
                    str(raw_path), str(ROOT/'calibration_predictions.npz')], check=True, cwd='/tmp')
    # Fresh dependency environment, with CPU wheels, independent of research site-packages.
    clean = ROOT / 'clean_env'
    subprocess.run(['uv', 'venv', '--python', sys.executable, str(clean)], check=True)
    clean_python = str(clean / 'bin/python')
    subprocess.run(['uv', 'pip', 'install', '--python', clean_python,
                    '--index-strategy', 'unsafe-best-match', '-r', str(STAGE/'requirements.txt')], check=True)
    subprocess.run([clean_python, '-I', str(ROOT/'verify_runtime.py'), str(STAGE),
                    str(raw_path), str(ROOT/'calibration_predictions.npz')], check=True, cwd='/tmp')
    # Exercise the actual CSV contract with a day whose observations are absent from history.
    fixture = ROOT / 'fixture'; fixture.mkdir(exist_ok=True)
    day = dates[-1]; h = raw['h'][-1]
    rows = [(day, zone, hour+1, h[z, hour]) for z, zone in enumerate(q.p.read_targets.__globals__['ZONES']) for hour in range(24)]
    pd.DataFrame(rows, columns=['delivery_date', 'market_id', 'hour_index', 'hourly_price_eur_mwh']).to_csv(fixture/'hourly.csv', index=False)
    original = pd.read_csv(source)
    original[original.delivery_date < day].to_csv(fixture/'history.csv', index=False)
    subprocess.run([clean_python, '-I', str(STAGE/'run.py'), '--hourly', str(fixture/'hourly.csv'),
                    '--history', str(fixture/'history.csv'), '--output', str(fixture/'predictions.csv'),
                    '--quantiles-output', str(fixture/'quantiles.csv')], check=True, cwd='/tmp')
    frame = pd.read_csv(fixture/'predictions.csv')
    assert len(frame) == 384 and not frame.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
    means = frame.groupby(['market_id','hour_index']).predicted_price_eur_mwh.mean().to_numpy().reshape(4,24)
    hourly_error = float(np.abs(means-h).max()); assert hourly_error <= .0001
    q.p.write(ROOT/'audit.json', dict(passed=True, training_days=348, calibration_days=15,
        fixed_budget_no_calibration_stopping=True, official_sha256=sha(source),
        causal_probes=[str(dates[i]) for i in [0,347,348,362]],
        isolated_cpu_replay_passed=True, csv_contract_passed=True, max_csv_hourly_mean_error=hourly_error,
        future_coverage_verified=False, clean_dependency_install_verified=True))
    environment = subprocess.check_output([clean_python, '-c',
        'import sys,torch,numpy,pandas,json;print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__)))'], text=True)
    q.p.write(STAGE/'environment.json', json.loads(environment))
    card = (STAGE/'SPECIFICATION_CARD.md').read_text()
    card = card.replace('**Status:** refit and packaging requested; completion requires the campaign\'s passing audit and completion receipt. This card describes the frozen recipe, not a claim that training has already finished.',
        '**Status:** fresh fitting, independent calibration, isolated CPU replay, clean dependency installation and CSV contract checks passed. See the matching audit and completion receipts.')
    (STAGE/'SPECIFICATION_CARD.md').write_text(card)
    shutil.copy2(ROOT/'audit.json', STAGE/'audit.json')
    hashes = {str(path.relative_to(STAGE)): sha(path) for path in sorted(STAGE.rglob('*')) if path.is_file() and '__pycache__' not in str(path)}
    q.p.write(STAGE/'MANIFEST.json', hashes)
    archive = ROOT/'champion_submission.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(STAGE, arcname='champion_submission', filter=lambda item: None if '__pycache__' in item.name else item)
    q.p.write(ROOT/'completion.json', dict(complete=True, audit_passed=True, archive=str(archive),
                                         sha256=sha(archive), bytes=archive.stat().st_size))
    print('REFIT AND PACKAGING COMPLETE', flush=True)


if __name__ == '__main__':
    run()
