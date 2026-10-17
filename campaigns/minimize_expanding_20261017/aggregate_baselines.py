"""Complete small-table reporting for corrected and raw interpolation denominators."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'results/minimize_expanding_20261017'


def read(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    audit = json.loads((ROOT / 'audit.json').read_text())
    assert audit['passed'] and audit['folds'] == 8
    original = {(r['month'], r['recipe']): r for r in read('monthly_scores.csv')}
    groups = {}
    for row in read('monthly_zone_scores.csv'):
        groups.setdefault((row['month'], row['recipe']), []).append(row)
    monthly = []
    assert set(groups) == set(original)
    for (month, recipe), zones in sorted(groups.items()):
        assert sorted(r['zone'] for r in zones) == [f'ZONE_{z}' for z in range(4)]
        # Every zone has the same dates/hours/quarters in this protocol.
        mae = sum(float(r['mae']) for r in zones)/4
        corrected = sum(float(r['interpolation_mae']) for r in zones)/4
        raw = sum(float(r['uncorrected_linear_interpolation_mae']) for r in zones)/4
        assert corrected > 0 and raw > 0, 'Undefined baseline ratio; do not smooth denominator'
        saved = original[month, recipe]
        assert abs(mae-float(saved['mae'])) < 1e-10
        assert abs(mae/corrected-float(saved['ratio'])) < 1e-10
        monthly.append(dict(month=month, recipe=recipe, test_days=int(saved['test_days']),
                            mae=mae, corrected_interpolation_mae=corrected,
                            corrected_interpolation_ratio=mae/corrected,
                            uncorrected_linear_interpolation_mae=raw,
                            uncorrected_linear_interpolation_ratio=mae/raw))
    write('alternate_interpolation_monthly_scores.csv', monthly)
    summaries = []
    for row in read('summary.csv'):
        group = [r for r in monthly if r['recipe'] == row['recipe']]
        assert len(group) == 8
        days = sum(r['test_days'] for r in group); assert days == 241
        primary = sum(r['corrected_interpolation_ratio'] for r in group)/8
        pooled = sum(r['mae']*r['test_days'] for r in group)/days
        cb = sum(r['corrected_interpolation_mae']*r['test_days'] for r in group)/days
        rb = sum(r['uncorrected_linear_interpolation_mae']*r['test_days'] for r in group)/days
        assert abs(primary-float(row['mean_monthly_ratio'])) < 1e-10
        assert abs(pooled-float(row['pooled_mae'])) < 1e-10
        summaries.append(dict(**row, pooled_corrected_interpolation_mae=cb,
                              pooled_corrected_interpolation_ratio=pooled/cb,
                              mean_monthly_uncorrected_linear_interpolation_ratio=
                                  sum(r['uncorrected_linear_interpolation_ratio'] for r in group)/8,
                              pooled_uncorrected_linear_interpolation_mae=rb,
                              pooled_uncorrected_linear_interpolation_ratio=pooled/rb))
    write('aggregate_comparison.csv', summaries)
    selected = next(r for r in summaries if r['recipe'] == 'validation_selector')
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
