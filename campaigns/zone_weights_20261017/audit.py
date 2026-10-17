"""Independent replay of frozen weights, calibration, monthly scores and sources."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zone_blend as e
import numpy as np
import pandas as pd


def audit(smoke=False):
    out = e.OUT/'smoke' if smoke else e.OUT
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['smoke'] == smoke
    assert manifest['source_manifest_sha256'] == e.sha(e.SOURCE/'manifest.json')
    assert manifest['source_audit_sha256'] == e.sha(e.SOURCE/'audit.json')
    assert manifest['quantile_helper_sha256'] == e.sha(e.PROJECT/'campaigns/monthly_quantiles_20261017/trainq.py')
    assert e.sha(e.PROJECT/'train_prices_15min.csv') == manifest['source_data_sha256']
    for path, digest in manifest['code_hashes'].items():
        assert e.sha(e.PROJECT/path) == digest, path
    for source in json.loads((out/'source_files.json').read_text()):
        assert e.sha(e.PROJECT/source['path']) == source['sha256'], source['path']
    lp = json.loads((out/'optimization_audit.json').read_text())
    assert lp['passed'] and len(lp['independent_exact_lp_checks']) == 10
    assert lp['maximum_gap'] <= .0005
    rows, interval_rows, counts = [], [], []
    maximum_mean = maximum_replay = 0.
    fold_paths = sorted((out/'folds').iterdir())
    assert len(fold_paths) == (1 if smoke else 8)
    for folder in fold_paths:
        source = e.SOURCE/'folds'/folder.name
        data = np.load(source/'data.npz')
        nt, ns, nc = (int(data[key]) for key in ['nt', 'ns', 'nc'])
        dates = data['dates'].astype(str)
        selection = json.loads((source/'selection.json').read_text())['by_recipe']
        decisions = json.loads((folder/'selection.json').read_text())
        report = json.loads((folder/'metrics.json').read_text())
        assert decisions['fit_dates'] == dates[nt:nt+ns].tolist()
        assert decisions['calibration_dates'] == dates[nt+ns:nt+ns+nc].tolist()
        assert all(date[:7] < folder.name for date in decisions['fit_dates']+decisions['calibration_dates'])
        assert all(date[:7] == folder.name for date in dates[nt+ns+nc:])
        bank = {}
        for mode in ['uniform', 'ratio']:
            names = [family+'_'+mode for family in e.FAMILIES]
            bank[mode] = np.stack([np.load(source/'trials'/str(selection[name]['config']['id'])/'quantiles.npy') for name in names])
            for fit in [decisions['optimization'][mode], decisions['tuning'][mode]['fit']]:
                for stage in ['global_fit', 'zone_fit']:
                    assert all(item['success'] for item in fit[stage]['stages'])
                    assert fit[stage]['device'].startswith('cuda')
        candidates = {name: item for name, item in decisions['policies'].items()
                      if 'inner_mae' in item and name != 'pretest_selector'}
        chosen = min(candidates, key=lambda name: (candidates[name]['inner_mae'], name))
        assert decisions['policies']['pretest_selector']['chosen_recipe'] == chosen
        truth = data['y'][nt+ns+nc:]
        baseline = np.abs(data['b'][nt+ns+nc:]-truth).mean()
        for recipe, policy in decisions['policies'].items():
            weights = np.asarray(policy['weights'])
            assert weights.shape == (4, 6)
            assert weights.min() >= 0 and np.max(abs(weights.sum(1)-1)) < 1e-12
            if recipe.startswith('global_'):
                assert np.abs(weights-weights[0]).max() < 1e-12
            # Independent accumulation, rather than the fitting script's einsum.
            members = bank[policy['mode']]
            whole = sum(members[index]*weights[None, :, None, None, index, None]
                        for index in range(6))
            test_q = whole[ns+nc:]
            saved = np.load(folder/'selected'/recipe/'predictions.npz')
            np.testing.assert_array_equal(saved['dates'], data['dates'][nt+ns+nc:])
            np.testing.assert_array_equal(saved['observed'], truth)
            assert np.isfinite(saved['quantiles']).all()
            assert np.diff(saved['quantiles'], axis=-1).min() >= -1e-9
            delta = float(abs(saved['quantiles']-test_q).max())
            assert delta < 1e-9
            maximum_replay = max(maximum_replay, delta)
            mean_error = float(abs(saved['quantiles'][..., 5].mean(-1)-data['h'][nt+ns+nc:]).max())
            assert mean_error < 1e-8
            maximum_mean = max(maximum_mean, mean_error)
            cal_q, cal_y = whole[ns:ns+nc], data['y'][nt+ns:nt+ns+nc]
            bands = e.q.intervals(cal_q, cal_y, test_q)
            calibration = json.loads((folder/'selected'/recipe/'calibration.json').read_text())
            previous_lower = previous_upper = None
            for coverage, lo, hi in e.q.INTERVALS:
                level = round(coverage*100)
                lower, upper = saved[f'lower_{level}'], saved[f'upper_{level}']
                np.testing.assert_allclose(lower, bands[level]['lower'], rtol=0, atol=1e-9)
                np.testing.assert_allclose(upper, bands[level]['upper'], rtol=0, atol=1e-9)
                np.testing.assert_allclose(calibration[str(level)]['correction'], bands[level]['correction'], rtol=0, atol=1e-9)
                assert (lower <= saved['quantiles'][..., 5]+1e-9).all()
                assert (upper >= saved['quantiles'][..., 5]-1e-9).all()
                if previous_lower is not None:
                    assert (lower <= previous_lower+1e-9).all() and (upper >= previous_upper-1e-9).all()
                previous_lower, previous_upper = lower, upper
                measured_coverage = float(((truth >= lower) & (truth <= upper)).mean())
                metric = next(item for item in report['results'][recipe]['intervals']
                              if item['kind'] == 'calibrated' and item['nominal'] == coverage)
                assert abs(metric['coverage']-measured_coverage) < 1e-12
                interval_rows.append(dict(month=folder.name, recipe=recipe, level=level, coverage=measured_coverage))
            pct = e.q.percentile(saved['upper_90']-saved['lower_90'], bands[90]['cal_upper']-bands[90]['cal_lower'])
            np.testing.assert_allclose(saved['uncertainty_percentile'], pct, rtol=0, atol=1e-12)
            expected = np.where(pct <= .25, 0, np.where(pct >= .75, 2, 1))
            np.testing.assert_array_equal(saved['confidence_bin'], expected)
            mae = float(abs(saved['quantiles'][..., 5]-truth).mean())
            score = report['results'][recipe]
            assert abs(score['mae']-mae) < 1e-10 and abs(score['ratio']-mae/baseline) < 1e-10
            rows.append(dict(month=folder.name, recipe=recipe, mae=mae, ratio=mae/baseline, days=len(truth)))
        counts.append(len(truth)*4*24*4)
    frame = pd.DataFrame(rows)
    summaries = pd.read_csv(out/'summary.csv').set_index('recipe')
    monthly = pd.read_csv(out/'monthly_scores.csv')
    for recipe, group in frame.groupby('recipe'):
        assert abs(summaries.loc[recipe, 'mean_monthly_ratio']-group.ratio.mean()) < 1e-10
        assert abs(summaries.loc[recipe, 'pooled_mae']-np.average(group.mae, weights=group.days)) < 1e-10
        for row in group.itertuples():
            exported = monthly[(monthly.recipe == recipe) & (monthly.month == row.month)].iloc[0]
            assert abs(exported.ratio-row.ratio) < 1e-10
    original = pd.read_csv(e.SOURCE/'summary.csv').set_index('recipe')
    for control in ['blend_uniform', 'blend_ratio', 'validation_selector', 'interpolation']:
        for key in ['mean_monthly_ratio', 'pooled_mae', 'coverage90', 'width90']:
            if not smoke:
                assert abs(summaries.loc['original_'+control, key]-original.loc[control, key]) < 1e-10
    csv = pd.read_csv(out/'predictions.csv.gz')
    assert len(csv) == sum(counts)
    assert not csv.duplicated(['delivery_date', 'market_id', 'hour_index', 'quarter_index']).any()
    qcols = [f'q{tau:g}' for tau in e.q.TAUS]
    assert np.isfinite(csv[qcols].to_numpy()).all() and np.diff(csv[qcols].to_numpy(), axis=-1).min() >= -1e-7
    receipt = dict(passed=True, folds=len(fold_paths), weighted_recipes_per_fold=15,
                   replays=len(rows), exported_selector_quarters=len(csv),
                   maximum_quantile_replay_error=maximum_replay, maximum_hourly_mean_error=maximum_mean,
                   maximum_exact_lp_gap=lp['maximum_gap'],
                   checks=['immutable official data/expert/source hashes', 'chronological fit/calibration/test dates',
                           'GPU solver success and independent exact first-fold LP checks',
                           'pretest selector parity', 'nonnegative simplex and all-quantile reconstruction',
                           'hourly q50 means and quantile ordering', 'reserved-data interval calibration replay/nesting',
                           'coverage/confidence labels', 'monthly ratio/pooled MAE/export identity', 'unchanged matched controls'])
    e.write(out/'audit.json', receipt)
    print('ZONE WEIGHTS AUDIT PASSED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    audit(parser.parse_args().smoke)
