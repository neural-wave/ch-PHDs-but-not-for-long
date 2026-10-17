"""Small, metadata-only interpretation; no fitting or prediction execution."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/interpretability_20261017'
MODEL = ROOT / 'results/linear_quantiles_vast/fixed_september_10m/model.json'
d = json.loads(MODEL.read_text())
labels = ['half_hour', 'first_half_quarters', 'second_half_quarters']
effects = []
for j, name in enumerate(d['feature_names']):
    a, u, v = [20 * d['weight'][k][j] for k in range(3)]
    effects.append(dict(feature=name, a=a, u=u, v=v,
                        q1=a+u, q2=a-u, q3=-a+v, q4=-a-v))
assert max(abs(sum(row[q] for q in ['q1','q2','q3','q4'])) for row in effects) < 1e-10
OUT.mkdir(exist_ok=True)
with (OUT / 'linear_effects.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(effects[0])); w.writeheader(); w.writerows(effects)
summary = {'scope': 'Fixed September 10-month standalone linear checkpoint; effects in EUR/MWh per training SD, before clipping. Not champion expert coefficients.',
           'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
           'top_coefficients': {label: sorted(effects, key=lambda r: abs(r[key]), reverse=True)[:8]
                                for label, key in zip(labels, ['a','u','v'])}}
for filename, metric in [('monthly_zone_scores.csv', 'zone'), ('confidence_bins.csv', 'label')]:
    path = ROOT / 'results/minimize_expanding_20261017' / filename
    groups = defaultdict(list)
    for r in csv.DictReader(path.open()):
        if r['recipe'] == 'fixed_blend_ratio':
            groups[r[metric]].append(r)
    assert groups and all(len(rows) == 8 for rows in groups.values())
    summary[metric] = {g: {'equal_month_ratio': sum(float(r['ratio']) for r in rows)/8,
                           'equal_month_mae': sum(float(r['mae']) for r in rows)/8}
                       for g, rows in groups.items()}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'top_coefficients'}, indent=2))
