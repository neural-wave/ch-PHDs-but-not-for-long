"""Add 50% calibration intervals to a completed frozen campaign without retraining."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from reconstruction import read_targets,ZONES
from hierarchical_models import conformal_quantile

p=argparse.ArgumentParser()
p.add_argument('--model-dir',required=True)
a=p.parse_args()
out=Path(a.model_dir)
report=json.loads((out/'metrics.json').read_text())
assert 'offline_point_and_interval_replay_max_error' in report
dates,y=read_targets('train_prices_15min.csv')
h=y.mean(-1)
pd.DataFrame([(dates[d],zone,hr+1,h[d,z,hr]) for d in range(317,333) for z,zone in enumerate(ZONES) for hr in range(24)],
    columns=['delivery_date','market_id','hour_index','hourly_price_eur_mwh']).to_csv(out/'qa_calibration_hourly.csv',index=False)
subprocess.run([sys.executable,'code/predict_hierarchical.py','--model-dir',str(out),'--hourly',str(out/'qa_calibration_hourly.csv'),
    '--history','train_prices_15min.csv','--output',str(out/'qa_calibration.csv'),'--device','cuda'],check=True)
calpred=pd.read_csv(out/'qa_calibration.csv').predicted_price_eur_mwh.to_numpy().reshape(16,4,24,4)
cal=np.abs(calpred-y[317:333])
with np.load(out/'test_predictions.npz') as loaded:
    arrays={k:loaded[k] for k in loaded.files}
pred=arrays['predicted']
testerr=np.abs(pred-y[333:])
report['conformal']=[r for r in report['conformal'] if r['nominal']!=.5]
for kind,scores,shape in [('marginal',cal.transpose(0,2,1,3).reshape(-1,4,4),(1,4,1,4)),
    ('hour_joint',cal.max(-1).transpose(0,2,1).reshape(-1,4),(1,4,1,1))]:
    radius=conformal_quantile(scores,.5).reshape(shape)
    covered=testerr<=radius
    report['conformal'].append(dict(kind=kind,nominal=.5,radius=radius.ravel().tolist(),coverage=float(covered.mean()),
        coverage_by_zone=covered.mean((0,2,3)).tolist(),hour_joint_coverage=float(covered.all(-1).mean()),mean_width=float(2*radius.mean())))
    arrays[kind+'_50_lower'],arrays[kind+'_50_upper']=pred-radius,pred+radius
radius=float(conformal_quantile(cal.max((1,2,3)),.5))
report['conformal'].append(dict(kind='day_joint',nominal=.5,calibration_blocks=16,radius=radius,
    day_joint_coverage=float((testerr<=radius).all((1,2,3)).mean()),mean_width=2*radius))
arrays['day_joint_50_lower'],arrays['day_joint_50_upper']=pred-radius,pred+radius
# Median absolute-error quantile defines a 50%-coverage interval; it is not a new point predictor.
report['added_50_percent']='held-out calibration absolute-error 0.5 quantile; point predictions unchanged'
(out/'metrics.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(out/'test_predictions.npz',**arrays)
subprocess.run([sys.executable,'code/predict_hierarchical.py','--model-dir',str(out),'--hourly',str(out/'qa_hourly.csv'),
    '--history','train_prices_15min.csv','--output',str(out/'qa_50_replay.csv'),'--device','cuda'],check=True)
replay=pd.read_csv(out/'qa_50_replay_intervals.csv')
for key in arrays:
    if key.endswith('_lower') or key.endswith('_upper'):
        assert np.max(np.abs(replay[key].to_numpy().reshape(30,4,24,4)-arrays[key]))<1e-4
assert np.array_equal(pred,arrays['predicted'])
print('PASS: frozen point forecast plus 50% marginal/hour/day intervals and complete interval replay',flush=True)
