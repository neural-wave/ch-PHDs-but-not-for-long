"""Independent eight-fold chronology, checkpoint and interval-metric audit."""
import argparse,json,hashlib,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train as t
import numpy as np
import pandas as pd
import torch


def main(smoke=False):
    t.setup(17);root=t.ROOT/('smoke' if smoke else 'folds');raw=np.load(t.ROOT/'raw.npz');manifest=json.loads((t.ROOT/'manifest.json').read_text())
    assert hashlib.sha256((t.PROJECT/'train_prices_15min.csv').read_bytes()).hexdigest()==t.SHA
    for path,sha in manifest['code_hashes'].items():assert hashlib.sha256((t.PROJECT/path).read_bytes()).hexdigest()==sha,path
    qq,rr=t.features(raw['dates'],raw['h'],raw['dates'],raw['y'])
    np.testing.assert_array_equal(rr,raw['r']);np.testing.assert_array_equal(qq,raw['q'])
    assert not raw['r'][:,-96:,4:8].any() and not raw['r'][:,-96:,9].any()
    np.testing.assert_array_equal(raw['r'][:,-96:,:4],np.repeat(raw['h'],4,axis=-1).transpose(0,2,1)/100)
    lookup={str(date):yy for date,yy in zip(raw['dates'],raw['y'])}
    for i,date in enumerate(raw['dates']):
        for block,lag in enumerate(range(7,0,-1)):
            yy=lookup.get(str((pd.Timestamp(str(date))-pd.Timedelta(days=lag)).date()));z=raw['r'][i,block*96:(block+1)*96]
            if yy is None:assert not z.any()
            else:
                np.testing.assert_array_equal(z[:,9],np.ones(96))
                np.testing.assert_array_equal(z[:,4:8],(yy-t.baseline(yy.mean(-1)[None])[0]).reshape(4,96).T/20)
    jobs=t.folds(raw['dates']);jobs=[jobs[0],jobs[-1]] if smoke else jobs
    ratios={};pooled={};total=0;exports=[]
    for fold in jobs:
        folder=root/fold['test'];d=np.load(folder/'data.npz');nt,ns,nc=(int(d[k]) for k in ['nt','ns','nc']);n=len(d['y'])
        assert n==nt+ns+nc+fold['test_days']
        dates=d['dates'];assert all(x[:7]<fold['validation'] for x in dates[:nt])
        assert all(x[:7]==fold['validation'] and int(x[8:10])<=15 for x in dates[nt:nt+ns])
        assert all(x[:7]==fold['validation'] and int(x[8:10])>=16 for x in dates[nt+ns:nt+ns+nc])
        assert all(x[:7]==fold['test'] for x in dates[nt+ns+nc:])
        for key in ['y','h','b','dates']:np.testing.assert_array_equal(d[key],raw[key][:n])
        for key,hist in [('q',False),('r',True)]:
            x,mu,sd=t.normalize(raw[key][:n],nt,hist)
            np.testing.assert_array_equal(d[key],x);np.testing.assert_array_equal(d['mu'+key],mu);np.testing.assert_array_equal(d['sd'+key],sd)
        s=json.loads((folder/'selection.json').read_text());m=json.loads((folder/'metrics.json').read_text())
        expected=[c for c in t.configs() if not smoke or c['id'] in [0,8]]
        assert sorted([r['config'] for r in s['candidates']],key=lambda r:r['id'])==expected
        # Replay all candidates, independently rescore validation selection.
        allq={}
        for row in s['candidates']:
            dest=folder/'trials'/str(row['config']['id']);ck=torch.load(dest/'model.pt',weights_only=False,map_location='cpu');assert ck['config']==row['config']
            net=t.QuantileSSM(ck['config'],d['q'].shape[-1],d['r'].shape[-1]).cuda().eval();net.load_state_dict(ck['state']);out=[]
            with torch.no_grad():
                for start in range(nt,n,32):out.append(net(torch.tensor(d['q'][start:start+32],device='cuda'),torch.tensor(d['r'][start:start+32],device='cuda')).cpu().numpy())
            q=t.evaluation.physical(np.concatenate(out),d['b'][nt:]);saved=np.load(dest/'quantiles.npy');np.testing.assert_allclose(q,saved,atol=1e-5,rtol=0)
            assert np.isfinite(saved).all() and np.diff(saved,axis=-1).min()>=-1e-8
            assert np.abs(saved[...,5].mean(-1)-d['h'][nt:]).max()<1e-8
            val=float(np.abs(saved[:ns,...,5]-d['y'][nt:nt+ns]).mean());assert abs(val-row['validation_mae'])<1e-8
            assert abs(val/float(d['selection_interpolation_mae'])-row['selection_ratio'])<1e-5
            curve=np.loadtxt(dest/'curves.csv',delimiter=',',skiprows=1);assert abs(curve[:,1].min()-row['selection_ratio'])<1e-8
            allq[row['config']['id']]=saved;total+=1
        for f in t.FAMILIES:
            expected=min([r for r in s['candidates'] if r['config']['family']==f],key=lambda r:(r['selection_ratio'],r['config']['id']))
            assert s['by_recipe'][f]==expected
        winner=min(t.FAMILIES,key=lambda f:(s['by_recipe'][f]['selection_ratio'],f));assert s['by_recipe']['validation_selector']['chosen_recipe']==winner
        truth=d['y'][nt+ns+nc:];base=d['b'][nt+ns+nc:];den=float(np.abs(base-truth).mean());assert den>0
        for recipe,row in s['by_recipe'].items():
            q=np.load(folder/'trials'/str(row['config']['id'])/'quantiles.npy');cal=q[ns:ns+nc];test=q[ns+nc:]
            saved=np.load(folder/'selected'/recipe/'predictions.npz');np.testing.assert_array_equal(saved['quantiles'],test)
            stats=m['results'][recipe];error=np.abs(test[...,5]-truth);mae=float(error.mean());ratio=mae/den
            assert abs(mae-stats['mae'])<1e-10 and abs(ratio-stats['ratio'])<1e-10
            pin=truth[...,None]-test;pb=np.maximum(t.TAUS*pin,(t.TAUS-1)*pin).mean((0,1,2,3));np.testing.assert_allclose(pb,stats['pinball_by_quantile'],rtol=0,atol=1e-10)
            previous_lo=previous_hi=None
            for coverage,lo,hi in t.evaluation.INTERVALS:
                level=round(coverage*100);scores=np.maximum.reduce([cal[...,lo]-d['y'][nt+ns:nt+ns+nc],d['y'][nt+ns:nt+ns+nc]-cal[...,hi],np.zeros_like(cal[...,lo])])
                group=scores.transpose(1,3,0,2).reshape(4,4,-1);rank=math.ceil((group.shape[-1]+1)*coverage);corr=np.sort(group,axis=-1)[...,rank-1][None,:,None,:]
                lower=test[...,lo]-corr;upper=test[...,hi]+corr
                if previous_lo is not None:lower=np.minimum(lower,previous_lo);upper=np.maximum(upper,previous_hi)
                np.testing.assert_allclose(lower,saved[f'lower_{level}'],rtol=0,atol=1e-10);np.testing.assert_allclose(upper,saved[f'upper_{level}'],rtol=0,atol=1e-10)
                reported=next(a for a in stats['intervals'] if a['kind']=='calibrated' and a['nominal']==coverage)
                assert abs(float(((truth>=lower)&(truth<=upper)).mean())-reported['coverage'])<1e-12
                assert abs(float((upper-lower).mean())-reported['width'])<1e-10
                previous_lo,previous_hi=lower,upper
            ratios.setdefault(recipe,[]).append(ratio);pooled.setdefault(recipe,[]).append(error.ravel())
        exports.append(pd.read_csv(folder/'predictions.csv.gz'))
    summary=json.loads(((t.ROOT/'smoke') if smoke else t.ROOT).joinpath('metrics.json').read_text())
    for row in summary['summary']:
        assert abs(row['mean_monthly_ratio']-np.mean(ratios[row['recipe']]))<1e-10
        assert abs(row['pooled_mae']-np.concatenate(pooled[row['recipe']]).mean())<1e-10
    assert total==(4 if smoke else 128)
    if not smoke:
        df=pd.read_csv(t.ROOT/'predictions.csv.gz');assert len(df)==92544 and not df.duplicated(['delivery_date','market_id','hour_index','quarter_index']).any()
        assert len(pd.concat(exports))==len(df)
        text=['# One-layer linear S4D frequency study','', 'Developmental backtests; full VALIDATION_PROTOCOL.md windows, fitting-only preprocessing, frozen test parameters. Primary score is arithmetic mean of eight monthly ratios.','', '| Recipe | Mean monthly ratio | Pooled MAE | Calibrated 90% coverage | Width |','|---|---:|---:|---:|---:|']
        for r in summary['summary']:text.append(f"| {r['recipe']} | {r['mean_monthly_ratio']:.6f} | {r['pooled_mae']:.6f} | {r['coverage90']:.2%} | {r['width90']:.3f} |")
        (t.ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    t.write((t.ROOT/'smoke' if smoke else t.ROOT)/'audit.json',dict(passed=True,folds=len(jobs),checkpoints=total,chronology=True,train_only_preprocessing=True,all_candidate_replay=True,calibration_rank_recomputed=True,mean_monthly_ratio_recomputed=True,calibrated_intervals=True))
    print('AUDIT PASSED',smoke,total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
