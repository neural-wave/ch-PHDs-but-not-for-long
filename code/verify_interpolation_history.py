"""Verify exact lag semantics, missing days, and target leakage invariance."""
import json
import os
from pathlib import Path
import numpy as np
from reconstruction import read_targets,baseline
from interpolation_history import interpolation_history
from interpretable_models import compact_features

dates,y=read_targets('train_prices_15min.csv');h=y.mean(-1);probe=333
query=dates[probe:probe+1]
x,names=interpolation_history(query,dates,y)
assert x.shape==(1,4,24,63) and len(names)==63 and np.isfinite(x).all()
for lag in range(1,8):
    expected=(y[probe-lag]-baseline(h[probe-lag:probe-lag+1])[0])/20
    np.testing.assert_array_equal(x[0,...,(lag-1)*9:(lag-1)*9+4],expected)
    np.testing.assert_array_equal(x[0,...,(lag-1)*9+4:(lag-1)*9+8],np.abs(expected))
    assert (x[0,...,(lag-1)*9+8]==1).all()
poison=y.copy();poison[probe:]=poison[probe:]*47-891
np.testing.assert_array_equal(x,interpolation_history(query,dates,poison)[0])
np.testing.assert_array_equal(x,interpolation_history(query,dates[:probe],y[:probe])[0])
keep=np.arange(len(dates))!=probe-3
missing=interpolation_history(query,dates[keep],y[keep])[0]
assert (missing[...,18:27]==0).all()
np.testing.assert_array_equal(missing[...,:18],x[...,:18])
np.testing.assert_array_equal(missing[...,27:],x[...,27:])
assert (interpolation_history(query,dates[:0],y[:0])[0]==0).all()
base,base_names=compact_features(query,h[probe:probe+1],dates,y,True)
extra,extra_names=compact_features(query,h[probe:probe+1],dates,y,True,7)
assert extra.shape==(1,4,24,90) and len(extra_names)==90
np.testing.assert_array_equal(extra[...,:27],base)
np.testing.assert_array_equal(extra[...,27:],x)
# Model input/output compatibility and hourly conservation for every family.
import torch
from simple_search import model,configs
for family in ['linear','additive','mlp','cnn']:
    cfg=dict(family=family,width=16,depth=1 if family!='cnn' else 3)
    net=model(cfg,90,np.zeros((3,90),dtype='float32')).eval()
    with torch.no_grad():
        for parameter in net.parameters():parameter.uniform_(-.01,.01)
        output=net(torch.tensor(extra,dtype=torch.float32)).numpy()
    assert output.shape==(1,4,24,4) and np.isfinite(output).all()
    assert np.abs(output.mean(-1)).max()<1e-6
assert len(configs())==224
os.environ['SIMPLE_SEARCH_FEATURES']='compact,compact7'
assert len(configs())==224 and {c['feature'] for c in configs()}=={'compact','compact7'}
report=dict(status='passed',features=90,added_channels=63,calendar_days=7,
            checks=['exact signed and absolute residuals','current/future perturbation','past-only equivalence',
                    'missing-calendar-date mask','empty history','unchanged compact prefix',
                    'four-family forward and zero-mean outputs','default and opt-in config grids'])
print(json.dumps(report,indent=2))
if os.environ.get('HISTORY_AUDIT_OUT'):
    Path(os.environ['HISTORY_AUDIT_OUT']).write_text(json.dumps(report,indent=2)+'\n')
