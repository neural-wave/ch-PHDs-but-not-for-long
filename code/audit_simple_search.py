"""Recompute all cached scores and constraints independently of training."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('results/simple_5090');report=json.loads((root/'metrics.json').read_text());data=np.load(root/'data.npz')
assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()==json.loads((root/'manifest.json').read_text())['sha256']
assert len(report['candidates'])==224
assert len({r['config']['id'] for r in report['candidates']})==224
maxmean=0
for row in report['candidates']:
    pred=np.load(root/'trials'/str(row['config']['id'])/'predictions.npy')
    assert pred.shape==(61,4,24,4) and np.isfinite(pred).all()
    error=np.abs(pred-data['y'][302:])
    assert abs(error[:31].mean()-row['validation_mae'])<1e-10
    assert abs(error[31:].mean()-row['september_mae'])<1e-10
    maxmean=max(maxmean,float(np.abs(pred.mean(-1)-data['h'][302:]).max()))
assert maxmean<1e-8
assert report['selected']['config']==min(report['candidates'],key=lambda r:r['validation_mae'])['config']
for family,selected in report['by_family'].items():
    assert selected['config']==min([r for r in report['candidates'] if r['config']['family']==family],key=lambda r:r['validation_mae'])['config']
audit=dict(passed=True,trials=224,score_recomputation='all August and September scores agree within 1e-10',
           hourly_mean_max_error=maxmean,selection='all global/family selections match August argmin',
           prediction_shape=[61,4,24,4],source_hash='matches official manifest')
(root/'independent_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
