"""Posthoc August diagnostics; preserve original forecasts, thresholds and selection."""
import json
from pathlib import Path
import numpy as np

root=Path('results/quantile_2d_cnn_20261017')
data=np.load(root/'data.npz')
risk=np.load(root/'risk_predictions.npz')
confidence=np.load(root/'confidence_predictions.npz')
forecast=np.load(root/'trials/23/predictions.npz')
errors=np.abs(forecast['point']-data['y'][302:])
daily=errors.mean((1,2,3));dates=risk['dates']
np.testing.assert_allclose(daily,risk['actual_mae'],rtol=0,atol=1e-12)
np.testing.assert_array_equal(dates,data['dates'][302:])
width=(forecast['quantiles'][...,9]-forecast['quantiles'][...,1]).mean((1,2,3))
rows={}
for name,start,end in [('August1-15_selection',0,15),('August16-31_calibration',15,31),('September_development',31,61)]:
    e=daily[start:end];worst=start+int(e.argmax())
    predicted=confidence['inputs_uncertainty'][start:end]
    order=np.argsort(-predicted,kind='stable')
    top=order[:max(1,int(np.ceil(.1*len(e))))]
    rows[name]=dict(days=len(e),mean_mae=float(e.mean()),max_day_mae=float(e.max()),
        worst_date=str(dates[worst]),catastrophic_days=int((e>10).sum()),
        primary_probability_min=float(risk['inputs'][start:end].min()),
        primary_probability_max=float(risk['inputs'][start:end].max()),
        worst_day_predicted_mae=float(confidence['inputs_uncertainty'][worst]),
        worst_day_predicted_rank=int(np.where(order==worst-start)[0][0]+1),
        top10percent_actual_mae=float(e[top].mean()),
        interval_width90_actual_mae_pearson=float(np.corrcoef(width[start:end],e)[0,1]),
        maximum_zone_day_mae=float(errors[start:end].mean((2,3)).max()),
        maximum_quarter_absolute_error=float(errors[start:end].max()))
result=dict(model_trial=23,primary_event='joint four-zone dailyMAE>10EUR/MWh',
    continuous_probe='inputs_uncertainty',rank_convention='1 is highest predicted daily MAE',by_period=rows,
    caveat='August1-15 selects the CNN; August16-31 calibrates the probes. Their rankings are diagnostic, not independent evidence of generalization. September worst-day ranking demonstrates a miss. No thresholds or forecasts changed.')
(root/'validation_risk_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
