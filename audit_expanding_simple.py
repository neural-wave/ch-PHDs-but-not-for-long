"""Independent audit of cached fold scores, selections and conformal bounds."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd

def audit(root,smoke=False):
    reports=json.loads((root/'metrics.json').read_text())['folds'];assert len(reports)==(2 if smoke else 16)
    checked=0;maxmean=0
    for report in reports:
        meta=report['manifest'];folder=root/(meta['plan']+'_'+str(meta['months'])+'m');data=np.load(folder/'data.npz')
        nt=int(data['train_end']);ns=int(data['select_end'])-nt;nc=int(data['test_start'])-nt-ns
        assert 1<=nc<=16 and ns==15
        assert len(report['candidates'])==(10 if smoke else 230)
        for row in report['candidates']:
            pred=np.load(folder/'trials'/str(row['config']['id'])/'predictions.npy')
            assert np.isfinite(pred).all()
            maxmean=max(maxmean,float(np.abs(pred.mean(-1)-data['h'][nt:]).max()))
            assert abs(np.abs(pred[:ns]-data['y'][nt:nt+ns]).mean()-row['validation_mae'])<1e-10
            checked+=1
        for family,selected in report['selected'].items():
            assert selected['config']==min([r for r in report['candidates'] if r['config']['family']==family],key=lambda r:r['validation_mae'])['config']
            p=np.load(folder/(family+'_test.npz'));err=np.abs(p['predicted']-p['observed']);cal=p['calibration_error']
            assert abs(err.mean()-selected['test_mae'])<1e-10
            csv=pd.read_csv(folder/(family+'_intervals.csv'));assert len(csv)==len(err)*384
            assert not csv.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
            for c in selected['intervals']:
                kind=c['kind'];level=round(c['nominal']*100)
                score=(cal.transpose(0,2,1,3).reshape(-1,4,4) if kind=='marginal' else cal.max(-1).transpose(0,2,1).reshape(-1,4) if kind=='hour_joint' else cal.max((1,2,3)))
                rank=math.ceil((len(score)+1)*c['nominal']);assert rank==c['rank'] and len(score)==c['n']
                q=np.sort(score,axis=0)[rank-1] if rank<=len(score) else np.full(score.shape[1:],np.inf)
                rr=q[None,:,None,:] if kind=='marginal' else q[None,:,None,None] if kind=='hour_joint' else q.reshape(1,1,1,1)
                covered=err<=rr;coverage=(covered if kind=='marginal' else covered.all(-1) if kind=='hour_joint' else covered.all((1,2,3))).mean()
                assert abs(coverage-c['coverage'])<1e-12
                if c['finite']:
                    assert np.allclose(p[f'lower_{kind}_{level}'],p['predicted']-rr,atol=1e-10,rtol=0)
                    assert np.allclose(p[f'upper_{kind}_{level}'],p['predicted']+rr,atol=1e-10,rtol=0)
                    assert abs(2*q.mean()-c['width'])<1e-10
                else:assert kind=='day_joint' and level==95 and np.isinf(p[f'upper_{kind}_{level}']).all()
    assert maxmean<1e-8
    result=dict(passed=True,folds=len(reports),candidates_recomputed=checked,max_hourly_mean_error=maxmean,
                checks='all candidate selection scores; family argmin; test MAE; unique CSV rows; corrected conformal ranks, bounds, coverage and widths; 95% day intervals correctly infinite')
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();audit(Path('results/expanding_simple_5090'+('_smoke' if a.smoke else '')),a.smoke)
