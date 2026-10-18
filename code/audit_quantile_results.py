"""Independent saved-array audit of all16 completed quantile evaluation folds."""
import json
from pathlib import Path
import numpy as np
from reconstruction import read_targets

root=Path('results/linear_quantiles_vast');report=json.loads((root/'metrics.json').read_text())
dates,y=read_targets('train_prices_15min.csv');truth={str(d):v for d,v in zip(dates,y)}
max_mean_error=0.;max_metric_delta=0.;cases=0
for fold in report['folds']:
    folder=root/(fold['plan']+'_'+str(fold['training_months'])+'m')
    with np.load(folder/'test_predictions.npz') as a:
        ds=a['dates'];observed=a['observed'];predicted=a['predicted'];q=a['quantiles']
        assert len(set(ds))==len(ds)==fold['test_days'] and all(str(d).startswith(fold['test']) for d in ds)
        assert np.array_equal(observed,np.array([truth[str(d)] for d in ds]))
        assert q.shape==observed.shape+(11,) and np.isfinite(q).all() and np.diff(q,axis=-1).min()>=-1e-10
        assert np.array_equal(predicted,q[...,5])
        max_mean_error=max(max_mean_error,float(np.abs(predicted.mean(-1)-observed.mean(-1)).max()))
        max_metric_delta=max(max_metric_delta,abs(float(np.abs(predicted-observed).mean())-fold['test_metrics']['mae']))
        for c in fold['conformal']:
            if c['method']=='halfhour_median_conformal':
                label='halfhour';yy=observed.reshape(-1,4,24,2,2).mean(-1)
            else:label=c['method'];yy=observed
            level=round(c['nominal']*100);lo=a[f'{label}_{level}_lower'];hi=a[f'{label}_{level}_upper']
            assert np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all()
            cov=float(((yy>=lo)&(yy<=hi)).mean());width=float((hi-lo).mean())
            max_metric_delta=max(max_metric_delta,abs(cov-c['coverage']),abs(width-c['mean_width']))
            cases+=1
assert len(report['folds'])==16 and max_mean_error<1e-8 and max_metric_delta<1e-8
result=dict(status='passed',folds=16,interval_cases=cases,max_hourly_mean_error=max_mean_error,max_metric_difference=max_metric_delta,
    checks='saved test dates/truth identity,finite/order11quantiles,median identity,MAE,all saved interval coverage/widths,mean conservation')
(root/'independent_audit.json').write_text(json.dumps(result,indent=2));print(result)
