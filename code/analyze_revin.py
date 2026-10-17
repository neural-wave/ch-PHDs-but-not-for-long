import json
from pathlib import Path
import numpy as np
root=Path('results/monthly_revin_gpu1');r=json.loads((root/'metrics.json').read_text())
pairs=[]
cs=r['candidates'][2:]
for a,b in zip(cs[::2],cs[1::2]):
    assert a['config']['seed']==b['config']['seed'] and a['config']['blocks']==b['config']['blocks']
    pairs.append(dict(blocks=a['config']['blocks'],seed=a['config']['seed'],control_august_mae=a['validation_mae'],
        revin_august_mae=b['validation_mae'],revin_minus_control=b['validation_mae']-a['validation_mae']))
a=np.load(root/'test_predictions.npz');b=np.load(Path('results/monthly_graph_gpu1/test_predictions.npz'))
assert np.array_equal(a['dates'],b['dates']) and np.array_equal(a['observed'],b['observed'])
d=(np.abs(a['predicted']-a['observed'])-np.abs(b['predicted']-b['observed'])).mean((1,2,3))
rng=np.random.default_rng(17);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
summary=dict(pairs=pairs,matched_pairs_won_by_revin=sum(p['revin_minus_control']<0 for p in pairs),
    paired_difference_mean=float(np.mean([p['revin_minus_control'] for p in pairs])),
    selected_revin_component='revin_7',ensemble_september_mae_difference=float(d.mean()),
    ensemble_september_day_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),
    interpretation='Small ensemble gain with mixed matched-control results. Day-bootstrap interval is descriptive, not valid post-selection significance; September previously viewed.')
(root/'revin_analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
