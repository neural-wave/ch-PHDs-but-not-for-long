"""Rescore saved monthly backtests. No training or prediction changes.

Run with Python's standard library: python3 evaluate_normalized_validation.py
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/normalized_validation_20261017'
EXPECTED_SHA = 'f71ae6dbcca5f04e76e35840e29d064e69ad292f26d8ec7bf013bff2d904f65c'
MONTHS = [f'2026-{m:02d}' for m in range(2, 10)]
SOURCES = [
    'train_prices_15min.csv',
    'results/monthly_baseline_mae.csv',
    'results/expanding_simple_5090/learning_curves.csv',
    'results/expanding_simple_5090/metrics.json',
    'results/expanding_simple_5090/independent_audit.json',
    'results/expanding_vast/learning_curves.csv',
    'results/expanding_vast/metrics.json',
    'results/linear_quantiles_vast/learning_curves.csv',
    'results/linear_quantiles_vast/metrics.json',
    'results/linear_quantiles_vast/independent_audit.json',
]


def read_csv(path):
    with (ROOT / path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads((ROOT / path).read_text())


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (a, b)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def official_baselines():
    """Independent scalar implementation from the original official quarters."""
    groups = defaultdict(dict)
    for row in read_csv('train_prices_15min.csv'):
        key = (row['delivery_date'], row['market_id'])
        h, q = int(row['hour_index']) - 1, int(row['quarter_index']) - 1
        assert 0 <= h < 24 and 0 <= q < 4 and (h, q) not in groups[key]
        value = float(row['price_eur_mwh'])
        assert math.isfinite(value)
        groups[key][h, q] = value
    assert len(groups) == 363 * 4
    errors = defaultdict(list)
    max_mean_error = 0.0
    for (date, zone), values in sorted(groups.items()):
        assert len(values) == 96
        y = [values[h, q] for h in range(24) for q in range(4)]
        hourly = [fmean(y[h * 4:h * 4 + 4]) for h in range(24)]
        for h in range(24):
            raw = []
            for q in range(4):
                pos = h + (q + 0.5) / 4 - 0.5
                left = math.floor(pos)
                pred = (hourly[0] if pos <= 0 else hourly[-1] if pos >= 23
                        else hourly[left] * (1 - (pos - left)) + hourly[left + 1] * (pos - left))
                raw.append(pred)
            shift = hourly[h] - fmean(raw)
            corrected = [v + shift for v in raw]
            max_mean_error = max(max_mean_error, abs(fmean(corrected) - hourly[h]))
            for q in range(4):
                errors[date[:7], zone, 'corrected'].append(abs(y[h * 4 + q] - corrected[q]))
                errors[date[:7], zone, 'raw'].append(abs(y[h * 4 + q] - raw[q]))
    assert max_mean_error <= 0.0001
    baseline = {}
    for month in sorted({key[0] for key in errors}):
        baseline[month] = {'month': month}
        for kind in ['corrected', 'raw']:
            zones = [errors[month, f'ZONE_{z}', kind] for z in range(4)]
            assert len({len(v) for v in zones}) == 1
            baseline[month][kind] = fmean(fmean(v) for v in zones)
            for z, values in enumerate(zones):
                baseline[month][f'{kind}_ZONE_{z}'] = fmean(values)
        baseline[month]['days'] = len(zones[0]) // 96
        if baseline[month]['corrected'] <= 0 or baseline[month]['raw'] <= 0:
            raise ValueError(f'Undefined normalized score: zero interpolation MAE in {month}')
    assert sum(row['days'] for row in baseline.values()) == 363
    for row in read_csv('results/monthly_baseline_mae.csv'):
        close(baseline[row['month']]['corrected'], float(row['linear_interpolation_mean_corrected']))
        close(baseline[row['month']]['raw'], float(row['linear_interpolation']))
    return baseline, max_mean_error


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256((ROOT / SOURCES[0]).read_bytes()).hexdigest() == EXPECTED_SHA
    baseline, max_mean_error = official_baselines()
    simple = read_json('results/expanding_simple_5090/metrics.json')
    simple_audit = read_json('results/expanding_simple_5090/independent_audit.json')
    assert simple_audit['passed'] and simple_audit['folds'] == 16
    assert simple_audit['candidates_recomputed'] == 3680
    quantile = read_json('results/linear_quantiles_vast/metrics.json')
    quantile_audit = read_json('results/linear_quantiles_vast/independent_audit.json')
    assert quantile_audit['status'] == 'passed' and quantile_audit['folds'] == 16
    assert quantile['source_sha256'] == EXPECTED_SHA and not quantile['smoke']
    old = read_json('results/expanding_vast/metrics.json')
    assert old['source_sha256'] == EXPECTED_SHA and not old['smoke']
    assert old['audits'].startswith('passed:')
    indexed = {}
    for fold in simple['folds']:
        meta = fold['manifest']
        assert meta['sha256'] == EXPECTED_SHA
        assert meta['train_end'] < meta['validation'] < meta['test']
        assert meta['selection_days'] == 15
        for family, selected in fold['selected'].items():
            candidates = [r for r in fold['candidates'] if r['config']['family'] == family]
            winner = min(candidates, key=lambda r: r['validation_mae'])
            assert selected['config'] == winner['config']
            close(fmean(selected['daily_mae']), selected['test_mae'])
            close(fmean(selected['by_zone']), selected['test_mae'])
            indexed[meta['plan'], meta['months'], family] = (meta, selected)
    rows = []

    def append(row, method, source, zone_prefix='mae_ZONE_', score='test_mae'):
        month = row['test']; base = baseline[month]
        days = int(row.get('test_days', base['days']))
        assert days == base['days']
        mae = float(row[score]); zones = [float(row[f'{zone_prefix}{z}']) for z in range(4)]
        close(fmean(zones), mae)
        result = dict(plan='expanding', test_month=month, method=method,
                      train_days=int(row['train_days']), test_days=days, targets=days * 384,
                      mae=mae, interpolation_mae=base['corrected'],
                      mae_over_interpolation_mae=mae / base['corrected'],
                      raw_linear_interpolation_mae=base['raw'],
                      mae_over_raw_interpolation_mae=mae / base['raw'], source=source)
        for z in range(4):
            result[f'mae_ZONE_{z}'] = zones[z]
            result[f'interpolation_mae_ZONE_{z}'] = base[f'corrected_ZONE_{z}']
            result[f'ratio_ZONE_{z}'] = zones[z] / base[f'corrected_ZONE_{z}']
        rows.append(result)

    for row in read_csv('results/expanding_simple_5090/learning_curves.csv'):
        meta, selected = indexed[row['plan'], int(row['months']), row['family']]
        close(float(row['test_mae']), selected['test_mae'])
        assert row['test'] == meta['test'] and int(row['test_days']) == meta['test_days']
        for z in range(4):
            close(float(row[f'mae_ZONE_{z}']), selected['by_zone'][z])
        if row['family'] == 'interpolation':
            close(float(row['test_mae']), baseline[row['test']]['corrected'])
        if row['plan'] == 'expanding':
            append(row, row['family'], 'expanding_simple_5090')
    for row in read_csv('results/expanding_vast/learning_curves.csv'):
        if row['plan'] == 'expanding' and row['family'] in ['direct', 'hierarchical']:
            append(row, row['family'], 'expanding_vast')
    qindex = {(f['plan'], f['training_months']): f for f in quantile['folds']}
    for row in read_csv('results/linear_quantiles_vast/learning_curves.csv'):
        fold = qindex[row['plan'], int(row['training_months'])]
        close(float(row['median_mae']), fold['test_metrics']['mae'])
        assert row['test'] == fold['test']
        if row['plan'] == 'expanding':
            append(row, 'linear_quantile_median', 'linear_quantiles_vast', 'mae_zone', 'median_mae')
    methods = sorted({row['method'] for row in rows})
    assert len(rows) == 72 and len(methods) == 9
    assert len({(row['method'], row['test_month']) for row in rows}) == len(rows)
    for month in MONTHS:
        assert len({r['train_days'] for r in rows if r['test_month'] == month}) == 1
    summaries = []
    for method in methods:
        subset = sorted([row for row in rows if row['method'] == method], key=lambda r: r['test_month'])
        assert [row['test_month'] for row in subset] == MONTHS
        assert len({row['train_days'] for row in subset}) == 8
        total = sum(row['targets'] for row in subset)
        assert total == 241 * 384
        mae = sum(row['mae'] * row['targets'] for row in subset) / total
        interp = sum(row['interpolation_mae'] * row['targets'] for row in subset) / total
        ratio = fmean(row['mae_over_interpolation_mae'] for row in subset)
        summaries.append(dict(method=method, folds=8, test_days=241, targets=total,
                              pooled_mae=mae, mean_monthly_ratio=ratio,
                              improvement_percent=100 * (1 - ratio),
                              pooled_mae_over_pooled_interpolation_mae=mae / interp,
                              worst_monthly_ratio=max(row['mae_over_interpolation_mae'] for row in subset),
                              months_beating_interpolation=sum(row['mae_over_interpolation_mae'] < 1 - 1e-12 for row in subset)))
    summaries.sort(key=lambda r: r['mean_monthly_ratio'])
    rows.sort(key=lambda r: (r['test_month'], r['method']))
    write_csv('monthly_metrics.csv', rows)
    write_csv('summary.csv', summaries)
    write_csv('baseline_months.csv', [baseline[m] for m in sorted(baseline)])
    # Existing September-only studies remain a separate descriptive comparison.
    extra = []
    def september(method, mae, source, selection):
        SOURCES.append(source)
        extra.append(dict(method=method, mae=mae, interpolation_mae=baseline['2026-09']['corrected'],
                          mae_over_interpolation_mae=mae / baseline['2026-09']['corrected'],
                          selection=selection, scope='September only; no multi-month estimate', source=source))
    path = 'results/monthly_joint_blend_gpu1/metrics.json'
    september('monthly_joint_blend', read_json(path)['test']['mae'], path, 'Full August')
    path = 'results/parallel_2d_tuning_20261017/metrics.json'; d = read_json(path)
    september('analog90_CNN', d['by_arm'][d['selected_arm']]['september_mae'], path, 'August 1-15')
    path = 'results/quantile_2d_cnn_20261017/metrics.json'; d = read_json(path)
    for group in [d['selected_group'], d['quantile_group']]:
        september('quantile_campaign_' + group, d['by_group'][group]['september_mae'], path, 'August 1-15; overall/cohort choice')
    extra.sort(key=lambda r: r['mae'])
    write_csv('september_only.csv', extra)
    controls = dict(passed=True, official_source_sha256=EXPECTED_SHA,
                    raw_quarter_targets=363 * 384, monthly_baselines_recomputed=True,
                    corrected_baseline_max_hourly_mean_error=max_mean_error,
                    saved_csv_scores_match_owner_reports=True,
                    simple_selection_and_daily_scores_rechecked=True,
                    simple_saved_independent_audit=simple_audit,
                    quantile_saved_independent_audit=quantile_audit,
                    folds=8, methods=9, monthly_rows=len(rows), test_days=241,
                    metric='Arithmetic mean across eight monthly MAE / corrected-interpolation-MAE ratios',
                    note='Quick rescore of existing saved results; no new neural prediction replay or retraining. Fixed-September history sizes excluded from CV aggregation.',
                    sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in dict.fromkeys(SOURCES)})
    (OUT / 'audit.json').write_text(json.dumps(controls, indent=2) + '\n')
    report = ['# Monthly validation with interpolation normalization', '',
              'Verified 17 October 2026. Quick rescore of completed developmental backtests; no retraining.', '',
              'Protocol: [VALIDATION_PROTOCOL.md](../../VALIDATION_PROTOCOL.md). Primary score is the arithmetic mean of the eight monthly MAE / mean-corrected-linear-interpolation-MAE ratios. Raw MAE is pooled by target count. Lower is better. Ratio 1 is interpolation parity.', '',
              '| Method | Pooled MAE (EUR/MWh) | Mean monthly ratio | Reduction vs interpolation | Worst monthly ratio |',
              '|---|---:|---:|---:|---:|']
    for r in summaries:
        report.append(f"| {r['method']} | {r['pooled_mae']:.6f} | {r['mean_monthly_ratio']:.6f} | {r['improvement_percent']:.2f}% | {r['worst_monthly_ratio']:.6f} |")
    report += ['', '## Month-by-month normalized errors', '',
               '| Test month | Interpolation MAE | ' + ' | '.join(methods) + ' |',
               '|---|---:|' + '---:|' * len(methods)]
    for month in MONTHS:
        group = {r['method']: r for r in rows if r['test_month'] == month}
        report.append(f"| {month} | {baseline[month]['corrected']:.4f} | " + ' | '.join(f"{group[m]['mae_over_interpolation_mae']:.4f}" for m in methods) + ' |')
    report += ['', '## September-only studies', '',
               'These methods have different model-selection campaigns and no eight-month estimate here. Do not rank their single-month ratios against the multi-month averages above.', '',
               '| Method | September MAE (EUR/MWh) | September ratio | Selection |',
               '|---|---:|---:|---|']
    for r in extra:
        report.append(f"| {r['method']} | {r['mae']:.6f} | {r['mae_over_interpolation_mae']:.6f} | {r['selection']} |")
    report += ['', '## Scope and verification', '',
               'All nine methods cover the same eight test months and 241 days (92,544 quarter targets across four zones). Linear, additive, MLP, CNN, ridge and interpolation come from the completed 3,680-candidate expanding/fixed-history campaign on root@93.91.156.85:55340, /workspace/HACKATON/results/expanding_simple_5090. Only expanding folds enter this comparison. Direct/hierarchical neural models and linear quantiles come from their earlier eight-fold campaigns. Quantile selection uses its joint median-MAE/pinball objective; candidate budgets differ across campaigns, so this is a comparison of saved pipelines, not a controlled architecture ablation.', '',
               'Independently recomputed both interpolation baselines from all official quarters. Checked saved monthly baselines, source hash, fold dates/counts, score consistency, simple-family validation argmins, daily MAE averages, per-zone averages and completed independent-audit receipts. Existing audits contain prediction/conservation/replay evidence. This rescore does not rerun all model predictions.', '',
               'The denominator uses hour-centred linear interpolation with flat day-edge extension, followed by a constant correction within each hour to recover the supplied hourly mean. Uncorrected linear interpolation is also included as an explicitly named alternative denominator in monthly_metrics.csv. In September its MAE is 7.540344 versus 7.090321 after mean correction. Do not mix these denominators.', '',
               'The reported average is mean(monthly model MAE / same-month interpolation MAE). It differs from pooled model MAE / pooled interpolation MAE, which remains a secondary column in summary.csv. Monthly ratios normalize baseline difficulty; they do not remove all seasonal distribution shifts or provide independent fold replications.', '',
               'All months were available during previous development. Scores remain developmental. The newer analog and quantile CNNs have September-only evidence here. No full nested re-selection, new candidate fitting or deployment replacement was performed.', '',
               'Reproduce: `python3 evaluate_normalized_validation.py`. Inputs and SHA256 receipts are listed in audit.json.']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({'summary': summaries, 'september_only': extra, 'audit_passed': True}, indent=2))


if __name__ == '__main__':
    main()
