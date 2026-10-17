"""Exploratory continuous day-error probes when severe-day events are too sparse."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from quantile_cnn import ROOT, TRAIN_END, SELECT_END, CAL_END, write_json


def association(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:
        return dict(spearman=None,pearson=None)
    return dict(spearman=float(spearmanr(x,y).statistic),pearson=float(np.corrcoef(x,y)[0,1]))


def verify_confidence():
    oof=np.load(ROOT/'risk_oof.npz');future=np.load(ROOT/'risk_predictions.npz')
    report=json.loads((ROOT/'confidence_metrics.json').read_text())
    saved=np.load(ROOT/'confidence_predictions.npz')
    ca=slice(SELECT_END-TRAIN_END,CAL_END-TRAIN_END);te=slice(CAL_END-TRAIN_END,None)
    for name,model in report['probes'].items():
        count=len(model['weight']);x=oof['features'][:,:count];mu=x.mean(0);sd=x.std(0).clip(.05)
        np.testing.assert_array_equal(mu,model['mu']);np.testing.assert_array_equal(sd,model['sd'])
        z=np.clip((x-mu)/sd,-15,15);center=oof['actual_mae'].mean()
        w=np.linalg.solve(z.T@z+10*np.eye(count),z.T@(oof['actual_mae']-center))
        np.testing.assert_allclose(w,model['weight'],rtol=0,atol=1e-10)
        xx=np.clip((future['features'][:,:count]-mu)/sd,-15,15);pred=xx@w+center
        offset=np.sum(future['actual_mae'][ca]-pred[ca])/(CAL_END-SELECT_END+10)
        assert abs(offset-model['calibration_offset'])<1e-10
        pred=np.maximum(pred+offset,0)
        np.testing.assert_allclose(pred,saved[name],rtol=0,atol=1e-10)
        mae=float(np.abs(pred[te]-future['actual_mae'][te]).mean())
        assert abs(mae-model['september']['day_error_prediction_mae'])<1e-10
    assert oof['indices'].max()<TRAIN_END and report['ridge_penalty']==10
    return True


def main():
    oof=np.load(ROOT/'risk_oof.npz');future=np.load(ROOT/'risk_predictions.npz')
    ca=slice(SELECT_END-TRAIN_END,CAL_END-TRAIN_END);te=slice(CAL_END-TRAIN_END,None)
    x,y=oof['features'],oof['actual_mae'];fy=future['actual_mae']
    models={};predictions={};fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    baseline=float(fy[ca].mean())
    for (name,count),ax in zip([('inputs',11),('inputs_uncertainty',19),('plus_conv',23),('plus_fusion',27)],axes.flat):
        mu=x[:,:count].mean(0);sd=x[:,:count].std(0).clip(.05)
        z=np.clip((x[:,:count]-mu)/sd,-15,15);center=float(y.mean())
        weight=np.linalg.solve(z.T@z+10*np.eye(count),z.T@(y-center))
        xx=np.clip((future['features'][:,:count]-mu)/sd,-15,15);pred=xx@weight+center
        offset=float(np.sum(fy[ca]-pred[ca])/(CAL_END-SELECT_END+10))
        pred=np.maximum(pred+offset,0);pt=pred[te];yt=fy[te]
        order=np.argsort(pt);groups=np.array_split(order,5);top=np.argsort(-pt)[:3]
        stats=dict(**association(pt,yt),day_error_prediction_mae=float(np.abs(pt-yt).mean()),
                   top10percent_actual_mae=float(yt[top].mean()),all_day_actual_mae=float(yt.mean()),
                   quintiles=[dict(days=int(len(g)),predicted_mae=float(pt[g].mean()),actual_mae=float(yt[g].mean())) for g in groups])
        models[name]=dict(mu=mu.tolist(),sd=sd.tolist(),weight=weight.tolist(),bias=center,
                          calibration_offset=offset,september=stats)
        predictions[name]=pred
        ax.scatter(pt,yt,s=25);lo=min(float(pt.min()),float(yt.min()));hi=max(float(pt.max()),float(yt.max()))
        ax.plot([lo,hi],[lo,hi],'k:');ax.set(xlabel='Predicted daily reconstruction MAE',ylabel='Actual daily MAE',title=f"{name}; Spearman {stats['spearman']:.2f}")
    fig.savefig(ROOT/'confidence_day_error.png',dpi=150);plt.close(fig)
    result=dict(probes=models,ridge_penalty=10,training_days=len(y),calibration_days=16,
                constant_calibration_baseline=baseline,
                constant_baseline_september_mae=float(np.abs(fy[te]-baseline).mean()),
                width90_vs_actual_mae=association(future['features'][te,12],fy[te]),
                caveat='Exploratory continuous-error diagnostic added after the predefined10EUR/MWh severe-day test had only1 training event and0 September events. All four fixed probes reported; no model selection or reconstruction changes. Fit182 earlier held-out errors, offset on August16-31 only.30 Septemberdays cannot establish reliable catastrophe detection.')
    write_json(ROOT/'confidence_metrics.json',result)
    np.savez(ROOT/'confidence_predictions.npz',dates=future['dates'],actual_mae=fy,**predictions)
    assert verify_confidence()
    result['artifact_replay_passed']=True;write_json(ROOT/'confidence_metrics.json',result)
    lines=['','Exploratory continuous daily-error probes (the primary severe-day test remains unchanged):','',
           '| Features | Spearman with daily MAE | Error predicting daily MAE |','|---|---:|---:|']
    for name,model in models.items():
        r=model['september'];lines.append(f"| {name} | {r['spearman']:.3f} | {r['day_error_prediction_mae']:.3f} |")
    lines += ['',f"Constant prediction from calibration days: error {result['constant_baseline_september_mae']:.3f} EUR/MWh.",
              f"Raw90% interval width versus actual daily MAE: Spearman {result['width90_vs_actual_mae']['spearman']:.3f}.",'',result['caveat']]
    with (ROOT/'REPORT.md').open('a') as file:file.write('\n'.join(lines)+'\n')
    print('CONTINUOUS CONFIDENCE AUDIT PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
