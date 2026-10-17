"""Descriptive fixed-interpolation difficulty; no predictor fitting or selection."""
import json
from pathlib import Path
import numpy as np

root=Path('results/quantile_2d_cnn_20261017')
d=np.load(root/'data.npz');residual=d['y']-d['b']
daily=np.abs(residual).mean((1,2,3));h=d['h']
ramp=np.abs(np.diff(h,axis=-1));curvature=np.abs(np.diff(h,n=2,axis=-1))
rows={}
for name,start,end in [('training',0,302),('August_selection',302,317),('August_calibration',317,333),('September_development',333,363)]:
    error=daily[start:end];worst=start+int(error.argmax())
    rows[name]=dict(days=len(error),mean_baseline_mae=float(error.mean()),minimum_day_mae=float(error.min()),
        maximum_day_mae=float(error.max()),worst_date=str(d['dates'][worst]),days_above10=int((error>10).sum()),
        day_mae_quantiles=np.quantile(error,[.1,.5,.9]).tolist(),
        ramp_pearson=float(np.corrcoef(ramp[start:end].mean((1,2)),error)[0,1]),
        curvature_pearson=float(np.corrcoef(curvature[start:end].mean((1,2)),error)[0,1]))
result=dict(target='Daily MAE of fixed mean-corrected centered linear interpolation, joint four zones',
    by_period=rows,hourly_residual_sum_max=float(np.abs(residual.sum(-1)).max()),
    caveat='Descriptive associations only. No daily interpolation-error predictor fitted, no prediction accuracy established. September already developmental. The existing CNN predicts the signed residual curve, not this scalar absolute-error target.')
(root/'interpolation_difficulty_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
