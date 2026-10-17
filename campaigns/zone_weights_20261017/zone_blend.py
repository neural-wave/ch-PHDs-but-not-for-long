"""Fit posthoc convex quantile blends; optimization data never include test labels."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', 'GPU-0bd8ea24-b611-23b4-3e0a-66d00e3ae3a9')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'campaigns/monthly_quantiles_20261017'))
import trainq as q
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, eye, hstack, vstack

SOURCE = PROJECT / 'results/monthly_quantiles_20261017'
OUT = PROJECT / 'results/zone_weights_20261017'
FAMILIES = q.p.FAMILIES
ALPHAS = np.array([0., .25, .5, .75, 1.])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_simplex(weights):
    """Only repair tiny numerical constraint error from the bounded solver."""
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.)
    return weights / weights.sum(axis=-1, keepdims=True)


def fit_gpu(pred, target, initial):
    """pred: groups x examples x experts; target: groups x examples."""
    assert torch.cuda.is_available(), 'This experiment requires the authorized GPU'
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups, count, experts = pred.shape
    assert target.shape == (groups, count)
    initial = project_simplex(initial)
    assert initial.shape == (groups, experts)
    # Subtract target before contraction; sum-to-one makes this exactly the error.
    residuals = torch.as_tensor(pred-target[..., None], device='cuda', dtype=torch.float64)
    params = initial.ravel().copy()
    constraint_jac = np.kron(np.eye(groups), np.ones((1, experts)))
    constraint = dict(type='eq', fun=lambda w: w.reshape(groups, experts).sum(1)-1,
                      jac=lambda w: constraint_jac)
    stages = []
    for smooth in [.2, .02, .002, .0001]:
        def objective(values):
            w = torch.as_tensor(values.reshape(groups, experts), device='cuda', dtype=torch.float64)
            errors = (residuals*w[:, None, :]).sum(-1)
            distance = torch.sqrt(errors.square()+smooth*smooth)
            value = distance.mean()
            gradient = ((errors/distance)[..., None]*residuals).mean(1)/groups
            return float(value.cpu()), gradient.cpu().numpy().ravel()
        result = minimize(objective, params, jac=True, method='SLSQP',
                          bounds=[(0., 1.)]*len(params), constraints=[constraint],
                          options=dict(maxiter=1000, ftol=1e-12))
        assert result.success, (smooth, result.message)
        params = result.x.copy()
        stages.append(dict(smooth=smooth, iterations=int(result.nit), success=bool(result.success)))
    weights = project_simplex(params.reshape(groups, experts))
    initial_mae = np.abs(((pred-target[..., None])*initial[:, None, :]).sum(-1)).mean(1)
    final_mae = np.abs(((pred-target[..., None])*weights[:, None, :]).sum(-1)).mean(1)
    # Smoothing at the final stage can perturb the exact optimum by <=0.0001.
    assert np.all(final_mae <= initial_mae + .0002), (initial_mae, final_mae)
    return weights, dict(stages=stages, initial_mae=initial_mae.tolist(), fit_mae=final_mae.tolist(),
                         device=str(residuals.device), max_simplex_error=float(abs(weights.sum(1)-1).max()))


def weight_fit(medians, target, initial):
    """Only this explicitly supplied selection slice is accessible to the fit."""
    # input expert x date x zone x hour x quarter
    experts = len(medians)
    x = medians.transpose(1, 2, 3, 4, 0)
    global_w, global_log = fit_gpu(x.reshape(1, -1, experts), target.reshape(1, -1), initial[None, :])
    zones = x.transpose(1, 0, 2, 3, 4).reshape(4, -1, experts)
    truth = target.transpose(1, 0, 2, 3).reshape(4, -1)
    zone_w, zone_log = fit_gpu(zones, truth, np.repeat(global_w, 4, 0))
    return global_w[0], zone_w, dict(global_fit=global_log, zone_fit=zone_log)


def interpolate_weights(global_w, zone_w, alpha):
    return project_simplex((1-np.asarray(alpha))[:, None]*global_w[None, :] +
                           np.asarray(alpha)[:, None]*zone_w)


def mix_quantiles(members, weights):
    return np.einsum('ez,edzhqk->dzhqk', weights.T, members, optimize=True)


def tune(medians, target, initial):
    """No calibration or test arrays accepted; chronological pre-test tuning only."""
    assert len(target) == 15
    global_w, zone_w, log = weight_fit(medians[:, :10], target[:10], initial)
    errors = []
    for alpha in ALPHAS:
        w = interpolate_weights(global_w, zone_w, np.full(4, alpha))
        pred = np.einsum('ez,edzhq->dzhq', w.T, medians[:, 10:15], optimize=True)
        errors.append(np.abs(pred-target[10:15]).mean((0, 2, 3)))
    errors = np.asarray(errors)
    common_index = int(errors.mean(1).argmin())
    zone_indices = errors.argmin(0)
    return dict(common_alpha=float(ALPHAS[common_index]), zone_alphas=ALPHAS[zone_indices].tolist(),
                common_score=float(errors[common_index].mean()),
                zone_score=float(errors[zone_indices, np.arange(4)].mean()),
                candidate_zone_mae=errors.tolist(), fit=log)


def exact_lp(medians, target):
    """Independent CPU verification on the remote host, not used to pick weights."""
    x = medians.reshape(len(medians), -1).T
    y = target.ravel()
    count, experts = x.shape
    matrix = vstack([hstack([csr_matrix(x), -eye(count)]),
                     hstack([-csr_matrix(x), -eye(count)])], format='csr')
    equality = csr_matrix(np.r_[np.ones(experts), np.zeros(count)][None, :])
    result = linprog(np.r_[np.zeros(experts), np.ones(count)/count], A_ub=matrix,
                     b_ub=np.r_[y, -y], A_eq=equality, b_eq=[1.],
                     bounds=[(0., 1.)]*experts+[(0., None)]*count, method='highs')
    assert result.success, result.message
    return float(result.fun)


def metrics_and_save(folder, recipe, members, weights, data, nt, ns, nc, meta):
    blended = mix_quantiles(members, weights)
    assert np.isfinite(blended).all()
    assert np.diff(blended, axis=-1).min() >= -1e-9
    maximum = float(np.abs(blended[..., 5].mean(-1)-data['h'][nt:]).max())
    assert maximum < 1e-8
    truth = data['y'][nt+ns+nc:]
    base = data['b'][nt+ns+nc:]
    test_q = blended[ns+nc:]
    calibration_q = blended[ns:ns+nc]
    calibration_y = data['y'][nt+ns:nt+ns+nc]
    bands = q.intervals(calibration_q, calibration_y, test_q)
    width = bands[90]['upper']-bands[90]['lower']
    pct = q.percentile(width, bands[90]['cal_upper']-bands[90]['cal_lower'])
    labels = np.where(pct <= .25, 0, np.where(pct >= .75, 2, 1)).astype('int8')
    error = np.abs(test_q[..., 5]-truth)
    baseline = np.abs(base-truth)
    pb_error = truth[..., None]-test_q
    pinball = np.maximum(q.TAUS*pb_error, (q.TAUS-1)*pb_error).mean((0, 1, 2, 3))
    saved = dict(dates=data['dates'][nt+ns+nc:], observed=truth, quantiles=test_q, taus=q.TAUS,
                 weights=weights, uncertainty_percentile=pct, confidence_bin=labels)
    intervals = []
    calibration = {}
    for coverage, lo, hi in q.INTERVALS:
        level = round(100*coverage)
        item = bands[level]
        saved[f'lower_{level}'] = item['lower']
        saved[f'upper_{level}'] = item['upper']
        calibration[str(level)] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in item.items()
                                  if key not in ['lower', 'upper', 'cal_lower', 'cal_upper']}
        for kind, lower, upper in [('raw', test_q[..., lo], test_q[..., hi]),
                                   ('calibrated', item['lower'], item['upper'])]:
            covered = (truth >= lower) & (truth <= upper)
            intervals.append(dict(kind=kind, nominal=coverage, coverage=float(covered.mean()),
                                  width=float((upper-lower).mean()), by_zone=covered.mean((0, 2, 3)).tolist()))
    bins = []
    for index, label in enumerate(['narrow', 'middle', 'wide']):
        mask = labels == index
        count = int(mask.sum())
        bins.append(dict(label=label, n=count, mae=float(error[mask].mean()) if count else None,
                         interpolation_mae=float(baseline[mask].mean()) if count else None,
                         coverage90=float(((truth >= bands[90]['lower']) & (truth <= bands[90]['upper']))[mask].mean()) if count else None,
                         width90=float(width[mask].mean()) if count else None))
    dest = folder/'selected'/recipe
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'predictions.npz', **saved)
    write(dest/'calibration.json', calibration)
    score = dict(mae=float(error.mean()), interpolation_mae=float(baseline.mean()),
                 ratio=float(error.mean()/baseline.mean()), by_zone=error.mean((0, 2, 3)).tolist(),
                 interpolation_by_zone=baseline.mean((0, 2, 3)).tolist(),
                 mean_pinball=float(pinball.mean()), pinball_ratio=float(pinball.mean()/baseline.mean()),
                 pinball_by_quantile=pinball.tolist(), intervals=intervals, confidence_bins=bins,
                 max_hourly_mean_error=maximum)
    if recipe == 'pretest_selector':
        dd, zz, hh, qq = np.indices(truth.shape)
        columns = dict(delivery_date=saved['dates'][dd.ravel()],
                       market_id=np.array([f'ZONE_{z}' for z in range(4)])[zz.ravel()],
                       hour_index=hh.ravel()+1, quarter_index=qq.ravel()+1)
        for k, tau in enumerate(q.TAUS):
            columns[f'q{tau:g}'] = test_q[..., k].ravel()
        columns.update(lower_90=bands[90]['lower'].ravel(), upper_90=bands[90]['upper'].ravel(),
                       width_90=width.ravel(), uncertainty_percentile=pct.ravel(),
                       confidence_bin=np.array(['narrow', 'middle', 'wide'])[labels.ravel()])
        pd.DataFrame(columns).to_csv(folder/'predictions.csv.gz', index=False, float_format='%.8f')
    return score


def main(smoke=False):
    started = time.time()
    torch.set_num_threads(1)
    assert json.loads((SOURCE/'audit.json').read_text())['passed']
    out = OUT/'smoke' if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((SOURCE/'manifest.json').read_text())
    manifest = dict(source='results/monthly_quantiles_20261017', source_manifest_sha256=sha(SOURCE/'manifest.json'),
                    source_audit_sha256=sha(SOURCE/'audit.json'), source_data_sha256=old_manifest['source_sha256'],
                    code_hashes={str(p.relative_to(PROJECT)): sha(p) for p in [HERE/'zone_blend.py', HERE/'audit.py', HERE/'run.py', HERE/'PROTOCOL.md']},
                    quantile_helper_sha256=sha(PROJECT/'campaigns/monthly_quantiles_20261017/trainq.py'),
                    gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    fitting='preceding days1-15 only', tuning='fit1-10/score11-15; refit1-15',
                    calibration='preceding16-end', test='following full month', alpha_grid=ALPHAS.tolist(),
                    expert_order=FAMILIES, smoke=smoke)
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text()) == manifest, 'Use a new output path after a source change'
    write(out/'manifest.json', manifest)
    reports = []
    source_files = []
    lp_checks = []
    folds = sorted((SOURCE/'folds').iterdir())
    if smoke:
        folds = folds[:1]
    for fold_index, source in enumerate(folds):
        month = source.name
        data = np.load(source/'data.npz')
        nt, ns, nc = (int(data[key]) for key in ['nt', 'ns', 'nc'])
        assert ns == 15
        dates = data['dates'].astype(str)
        assert dates[nt+ns-1] < dates[nt+ns] <= dates[nt+ns+nc-1] < dates[nt+ns+nc]
        selection = json.loads((source/'selection.json').read_text())['by_recipe']
        old_report = json.loads((source/'metrics.json').read_text())
        folder = out/'folds'/month
        folder.mkdir(parents=True, exist_ok=True)
        source_files += [dict(path=str(p.relative_to(PROJECT)), sha256=sha(p))
                         for p in [source/'data.npz', source/'selection.json', source/'metrics.json']]
        bank, policies, logs, tuning = {}, {}, {}, {}
        for mode in ['uniform', 'ratio']:
            names = [family+'_'+mode for family in FAMILIES]
            trial_files = [source/'trials'/str(selection[name]['config']['id'])/'quantiles.npy' for name in names]
            members = np.stack([np.load(path) for path in trial_files])
            source_files += [dict(path=str(p.relative_to(PROJECT)), sha256=sha(p)) for p in trial_files]
            assert members.shape[1:] == (len(dates)-nt, 4, 24, 4, 11)
            bank[mode] = members
            medians = members[..., 5]
            fit_target = data['y'][nt:nt+ns].copy()
            initial = np.array([selection['blend_'+mode]['weights'].get(name, 0.) for name in names])
            tuning[mode] = tune(medians[:, :ns].copy(), fit_target.copy(), initial)
            global_w, zone_w, log = weight_fit(medians[:, :ns].copy(), fit_target.copy(), initial)
            logs[mode] = log
            for alpha in ALPHAS:
                label = 'global' if alpha == 0 else 'zone' if alpha == 1 else f'zone_fraction_{alpha:g}'
                policies[label+'_'+mode] = dict(mode=mode, weights=interpolate_weights(global_w, zone_w, np.full(4, alpha)).tolist(), alpha=[float(alpha)]*4)
            for policy, alpha in [('common_selected', [tuning[mode]['common_alpha']]*4),
                                  ('zone_selected', tuning[mode]['zone_alphas'])]:
                policies[policy+'_'+mode] = dict(mode=mode, weights=interpolate_weights(global_w, zone_w, alpha).tolist(),
                                               alpha=alpha, inner_mae=tuning[mode]['common_score' if policy == 'common_selected' else 'zone_score'])
            # Exact LP is independent of the smooth GPU optimizer and the test month.
            if fold_index == 0:
                for zone in [None, 0, 1, 2, 3]:
                    pfit = medians[:, :ns] if zone is None else medians[:, :ns, zone]
                    yfit = fit_target if zone is None else fit_target[:, zone]
                    chosen = global_w if zone is None else zone_w[zone]
                    gpu_mae = float(np.abs(np.tensordot(chosen, pfit, axes=1)-yfit).mean())
                    optimum = exact_lp(pfit, yfit)
                    gap = gpu_mae-optimum
                    assert -.000001 <= gap <= .0005, (mode, zone, gpu_mae, optimum)
                    lp_checks.append(dict(mode=mode, zone=zone, gpu_mae=gpu_mae, exact_mae=optimum, gap=gap))
        candidates = {name: row for name, row in policies.items() if 'inner_mae' in row}
        chosen = min(candidates, key=lambda name: (candidates[name]['inner_mae'], name))
        policies['pretest_selector'] = dict(policies[chosen], chosen_recipe=chosen)
        # Persist all policy decisions before exposing any test targets to scoring.
        write(folder/'selection.json', dict(policies=policies, tuning=tuning, optimization=logs,
                                            source_test_month=month, fit_dates=dates[nt:nt+ns].tolist(),
                                            calibration_dates=dates[nt+ns:nt+ns+nc].tolist()))
        scores = {}
        for recipe, item in policies.items():
            scores[recipe] = metrics_and_save(folder, recipe, bank[item['mode']], np.asarray(item['weights']),
                                              data, nt, ns, nc, old_report['manifest'])
        for control in ['blend_uniform', 'blend_ratio', 'validation_selector', 'interpolation']:
            scores['original_'+control] = old_report['results'][control]
        report = dict(manifest=old_report['manifest'], selected=policies, results=scores)
        write(folder/'metrics.json', report)
        reports.append(report)
        write(out/'progress.json', dict(completed_folds=len(reports), total_folds=len(folds), last_month=month,
                                       elapsed_seconds=time.time()-started))
        print('ZONE WEIGHTS FOLD COMPLETE', month, 'selector', chosen, 'ratio', scores['pretest_selector']['ratio'], flush=True)
    write(out/'source_files.json', source_files)
    write(out/'optimization_audit.json', dict(passed=True, independent_exact_lp_checks=lp_checks,
                                             maximum_gap=max(item['gap'] for item in lp_checks)))
    # Reuse the original aggregation definitions without changing source files.
    previous_root = q.ROOT
    q.ROOT = out
    q.summarize(reports)
    q.ROOT = previous_root
    zone_rows, weight_rows = [], []
    for report in reports:
        month = report['manifest']['test']
        for recipe, score in report['results'].items():
            denom = score.get('interpolation_by_zone', report['results']['global_uniform']['interpolation_by_zone'])
            for zone in range(4):
                zone_rows.append(dict(month=month, recipe=recipe, zone=f'ZONE_{zone}', mae=score['by_zone'][zone],
                                      interpolation_mae=denom[zone], ratio=score['by_zone'][zone]/denom[zone]))
        for recipe, item in report['selected'].items():
            for zone, weights in enumerate(item['weights']):
                for family, weight in zip(FAMILIES, weights):
                    weight_rows.append(dict(month=month, recipe=recipe, mode=item['mode'], zone=f'ZONE_{zone}',
                                            family=family, weight=weight, zone_fraction=item['alpha'][zone]))
    zones = pd.DataFrame(zone_rows)
    zones.to_csv(out/'monthly_zone_scores.csv', index=False)
    zones.groupby(['recipe', 'zone']).agg(mean_monthly_ratio=('ratio', 'mean'), mean_monthly_mae=('mae', 'mean')).reset_index().to_csv(out/'zone_summary.csv', index=False)
    pd.DataFrame(weight_rows).to_csv(out/'weights.csv', index=False)
    pd.concat([pd.read_csv(out/'folds'/report['manifest']['test']/'predictions.csv.gz') for report in reports], ignore_index=True).to_csv(out/'predictions.csv.gz', index=False, float_format='%.8f')
    summary = pd.read_csv(out/'summary.csv')
    lines = ['# Posthoc zone weight experiment', '',
             'Frozen six-expert quantile predictions; GPU-fitted convex weights, separate interval calibration, chronological next-month tests. Developmental results, not a blind country-transfer test.', '',
             '| Recipe | Mean monthly MAE/interpolation MAE | Pooled MAE | 90% coverage | 90% width |',
             '|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.recipe} | {row.mean_monthly_ratio:.6f} | {row.pooled_mae:.6f} | {100*row.coverage90:.2f}% | {row.width90:.3f} |')
    lines += ['', 'Global uses six shared weights; zone uses six weights separately for each zone. Fractions shrink zone weights toward the global fit. Selected fractions use fit days1–10 and tuning days11–15, then refit on all15. The pretest selector selects objective and common/per-zone policy before the next test month. Expert checkpoint selection already saw all15 days, so the inner tuning score is not independently nested.', '',
              'All quantile levels and all four quarters share a zone weight vector. The median preserves supplied hourly means; final intervals are calibrated separately on days16–end. Confidence bins summarize relative calibrated width, not correctness probabilities.', '',
              f'Independent first-fold linear-program check: maximum GPU fitting MAE excess {max(item["gap"] for item in lp_checks):.8f} EUR/MWh. Full constraint/calibration/source replay is in audit.json.', '',
              'Artifacts: monthly_scores.csv, monthly_zone_scores.csv, zone_summary.csv, weights.csv, interval_metrics.csv, confidence_bins.csv, predictions.csv.gz, folds/*/selected/*/predictions.npz. No production model was replaced.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('ZONE WEIGHTS FITTING COMPLETE', len(reports), 'folds', time.time()-started, 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args().smoke)
