"""Reconstruct official inputs, selection, forecasts, calibration and risk artifacts."""
import hashlib
import json

import numpy as np
import torch

from reconstruction import read_targets, baseline
from quantile_cnn import (ROOT, SHA, TAUS, TRAIN_END, SELECT_END, CAL_END, OOF_EDGES,
                          SEVERE_MAE, configs, make_features, normalize, layout,
                          load_model, predict, write_json, configure)
from analyze_quantile_cnn import risk_features


def main():
    configure()
    assert hashlib.sha256(open('train_prices_15min.csv','rb').read()).hexdigest()==SHA
    dates,y=read_targets('train_prices_15min.csv');d=np.load(ROOT/'data.npz')
    np.testing.assert_array_equal(y,d['y']);np.testing.assert_array_equal(dates,d['dates'])
    np.testing.assert_array_equal(y.mean(-1),d['h']);np.testing.assert_array_equal(baseline(y.mean(-1)),d['b'])
    s=json.loads((ROOT/'selection.json').read_text());m=json.loads((ROOT/'metrics.json').read_text())
    rows=s['candidates'];assert len(rows)==24 and {r['config']['id'] for r in rows}==set(range(24))
    assert sorted([r['config'] for r in rows],key=lambda a:a['id'])==configs()
    assert s['overall']['config']['id']==min(rows,key=lambda r:r['validation_mae'])['config']['id']
    assert s['quantile']['config']['id']==min([r for r in rows if r['config']['quantile_weight']>0],key=lambda r:r['validation_mae'])['config']['id']
    features={}
    for pack in ['raw','analog90']:
        raw=make_features(dates,d['h'],dates,y,pack);f=np.load(ROOT/f'{pack}.npz')
        np.testing.assert_array_equal(raw,f['raw'])
        np.testing.assert_array_equal(raw[:302].mean((0,2),keepdims=True),f['mu'])
        np.testing.assert_array_equal(raw[:302].std((0,2),keepdims=True).clip(.05),f['sd'])
        np.testing.assert_array_equal(normalize(raw,f['mu'],f['sd'],pack),f['x'])
        ch,aux=layout(pack);g=raw[:,:-aux].reshape(-1,ch,16,96);ng=f['x'][:,:-aux].reshape(-1,ch,16,96)
        assert not g[:,:4,14].any() and not ng[:,:,14].any()
        np.testing.assert_allclose(g[:,:4,15],np.repeat(d['h'],4,axis=-1)/100,rtol=1e-6)
        np.testing.assert_array_equal(ng[:,4],g[:,4])
        changed=y.copy();changed[333:]+=123456
        probe=make_features(dates[333:334],d['h'][333:334],dates,changed,pack)
        np.testing.assert_array_equal(probe,raw[333:334])
        features[pack]=True
    for row in rows:
        folder=ROOT/'trials'/str(row['config']['id'])
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert ck['fit_end']==TRAIN_END and ck['val_end']==SELECT_END
        curve=np.loadtxt(folder/'learning_curves.csv',delimiter=',',skiprows=1)
        assert abs(float(curve[:,4].min())-row['validation_mae'])<1e-10
        assert int(curve[np.argmin(curve[:,4]),0])==row['best_epoch']
        assert all(torch.isfinite(v).all() for v in ck['state'].values())
    replay_checks={}
    qm=json.loads((ROOT/'quantile_metrics.json').read_text())
    for key,row in s['by_group'].items():
        expected=min([r for r in rows if r['config']['pack']==row['config']['pack'] and
                      r['config']['quantile_weight']==row['config']['quantile_weight']],key=lambda r:r['validation_mae'])
        assert row['config']['id']==expected['config']['id']
        folder=ROOT/'trials'/str(row['config']['id']);saved=np.load(folder/'predictions.npz')
        replay=predict(folder,TRAIN_END,len(y))
        for name in ['point','quantiles','conv','fused']:
            np.testing.assert_allclose(saved[name],replay[name],atol=1e-7,rtol=0)
            assert np.isfinite(saved[name]).all()
        p,q=saved['point'],saved['quantiles'];mean_error=float(np.abs(p.mean(-1)-d['h'][TRAIN_END:]).max())
        assert mean_error<1e-8 and np.diff(q,axis=-1).min()>=-1e-7
        val=float(np.abs(p[:SELECT_END-TRAIN_END]-y[TRAIN_END:SELECT_END]).mean())
        test=float(np.abs(p[CAL_END-TRAIN_END:]-y[CAL_END:]).mean())
        assert abs(val-row['validation_mae'])<1e-4 and abs(test-m['by_group'][key]['september_mae'])<1e-10
        if row['config']['quantile_weight']:
            ca=slice(SELECT_END-TRAIN_END,CAL_END-TRAIN_END);te=slice(CAL_END-TRAIN_END,None);truth=y[TRAIN_END:]
            err=truth[te,...,None]-q[te]
            pb=np.maximum(TAUS*err,(TAUS-1)*err).mean((0,1,2,3))
            np.testing.assert_allclose(pb,qm['by_group'][key]['pinball_by_tau'],rtol=0,atol=1e-10)
            cal=np.load(folder/'calibrated_intervals.npz')
            for level,(lo,hi) in {'50':(4,6),'80':(2,8),'90':(1,9),'95':(0,10)}.items():
                scores=np.maximum.reduce([q[ca,...,lo]-truth[ca],truth[ca]-q[ca,...,hi],np.zeros_like(truth[ca])])
                vals=scores.transpose(1,3,0,2).reshape(4,4,-1);rank=int(np.ceil((vals.shape[-1]+1)*int(level)/100))-1
                offsets=np.sort(vals,axis=-1)[...,rank]
                np.testing.assert_allclose(offsets,qm['by_group'][key]['intervals'][level]['offsets'],rtol=0,atol=1e-10)
                np.testing.assert_array_equal(cal[f'lower_{level}'],q[te,...,lo]-offsets[None,:,None,:])
                np.testing.assert_array_equal(cal[f'upper_{level}'],q[te,...,hi]+offsets[None,:,None,:])
                hit=(y[CAL_END:]>=cal[f'lower_{level}'])&(y[CAL_END:]<=cal[f'upper_{level}'])
                assert abs(float(hit.mean())-qm['by_group'][key]['intervals'][level]['calibrated_coverage'])<1e-12
        replay_checks[key]=dict(selection_mae=val,september_mae=test,hourly_mean_error=mean_error)
    om=json.loads((ROOT/'oof_manifest.json').read_text());risk=np.load(ROOT/'risk_oof.npz');all_indices=[];all_x=[]
    for fold,(start,end) in zip(om['folds'],zip(OOF_EDGES[:-1],OOF_EDGES[1:])):
        assert fold['fit_end']==start-15 and fold['val_end']==start
        folder=ROOT/'oof'/str(fold['fold']);p=np.load(folder/'predictions.npz')
        assert int(p['start'])==start and int(p['end'])==end
        replay=predict(folder,start,end)
        np.testing.assert_array_equal(p['point'],replay['point'])
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        raw=np.load(ROOT/f"{ck['config']['pack']}.npz")['raw']
        np.testing.assert_array_equal(ck['mu'],raw[:start-15].mean((0,2),keepdims=True))
        error=np.abs(p['point']-y[start:end]).mean((1,2,3));indices=np.arange(start,end)
        mask=np.isin(risk['indices'],indices)
        np.testing.assert_array_equal(error,risk['actual_mae'][mask])
        x,names=risk_features(d,indices,p);all_x.append(x);all_indices.extend(indices)
    np.testing.assert_array_equal(all_indices,risk['indices']);np.testing.assert_array_equal(np.concatenate(all_x),risk['features'])
    assert len(all_indices)==182 and max(all_indices)<TRAIN_END
    np.testing.assert_array_equal(risk['severe'],risk['actual_mae']>SEVERE_MAE)
    rm=json.loads((ROOT/'risk_models.json').read_text());rp=np.load(ROOT/'risk_predictions.npz')
    folder=ROOT/'trials'/str(s['quantile']['config']['id']);p=np.load(folder/'predictions.npz')
    future,names=risk_features(d,np.arange(TRAIN_END,len(y)),p)
    np.testing.assert_array_equal(future,rp['features'])
    # Current/future quarter targets cannot change the risk features of a query day.
    changed={k:d[k].copy() for k in d.files};changed['y'][CAL_END:]+=999999
    probe,_=risk_features(changed,[CAL_END],{k:p[k][CAL_END-TRAIN_END:CAL_END-TRAIN_END+1] for k in p.files})
    np.testing.assert_array_equal(probe,future[CAL_END-TRAIN_END:CAL_END-TRAIN_END+1])
    for name,model in rm['models'].items():
        count=len(model['weight']);xx=np.clip((future[:,:count]-np.asarray(model['mu']))/np.asarray(model['sd']),-15,15)
        logit=xx@np.asarray(model['weight'])+model['bias']+model['calibration_intercept']
        probability=1/(1+np.exp(-logit));np.testing.assert_allclose(probability,rp[name],rtol=0,atol=1e-12)
        expected_threshold=float(np.quantile(probability[SELECT_END-TRAIN_END:CAL_END-TRAIN_END],.9))
        assert abs(expected_threshold-model['alert_threshold'])<1e-12
        assert abs(np.mean((probability[CAL_END-TRAIN_END:]-rp['severe'][CAL_END-TRAIN_END:])**2)-model['september']['brier'])<1e-12
    explanations=json.loads((ROOT/'explanations'/'summary.json').read_text())
    assert explanations['cases'] and len(explanations['ablations'])==8*len(explanations['cases'])
    for case in explanations['cases']:
        a=np.load(ROOT/'explanations'/f"{case['date']}_maps.npz")
        assert all(np.isfinite(a[k]).all() for k in a.files)
    audit=dict(passed=True,feature_checks=features,model_checks=replay_checks,
               checks=['official CSV hash and arrays','24 unique predeclared fits','train-only normalization',
                       'query quarter masking and known hourly rows','future-target perturbation',
                       'August1-15 checkpoint/seed/model selection','saved checkpoint replay','hourly means',
                       'ordered independent marginal quantiles','pinball and calibration reconstruction',
                       '182 chronologically held-out risk labels','risk target causality','serialized risk replay',
                       'separate August16-31 calibration','finite explanation artifacts'])
    write_json(ROOT/'audit.json',audit)
    with (ROOT/'REPORT.md').open('a') as file:file.write('\nIndependent artifact audit: **PASSED**.\n')
    print('FINAL AUDIT PASSED',audit,flush=True)


if __name__=='__main__':main()
