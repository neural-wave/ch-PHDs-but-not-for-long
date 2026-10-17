"""Rebuild inputs, verify selection/checkpoints and compute development scores."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from reconstruction import read_targets,baseline
from ssm_models import Model
from ssm_experiment import ROOT,SHA,features,normalize,configs,setup,write


def replay(folder,d):
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    q,mq,sq=normalize(d['q'],ck['fit_end']);r,mr,sr=normalize(d['r'],ck['fit_end'],True)
    for name,arr in [('muq',mq),('sdq',sq),('mur',mr),('sdr',sr)]:np.testing.assert_array_equal(ck[name],arr)
    net=Model(ck['config'],q.shape[-1],r.shape[-1]).cuda().eval();net.load_state_dict(ck['state'])
    assert all(torch.isfinite(p).all() for p in net.parameters())
    with torch.no_grad():
        out=[]
        for first in range(ck['fit_end'],ck['stop'],32):
            sl=slice(first,min(first+32,ck['stop']))
            out.append(net(torch.tensor(q[sl],device='cuda'),torch.tensor(r[sl],device='cuda')).cpu().numpy())
    raw=np.concatenate(out).astype('float64');pred=d['b'][ck['fit_end']:ck['stop']]+20*(raw-raw.mean(-1,keepdims=True))
    saved=np.load(folder/'predictions.npy');np.testing.assert_allclose(saved,pred,atol=1e-5,rtol=0)
    meanerr=float(np.abs(saved.mean(-1)-d['h'][ck['fit_end']:ck['stop']]).max());assert meanerr<1e-8
    m=json.loads((folder/'metrics.json').read_text());curve=np.loadtxt(folder/'curves.csv',delimiter=',')
    assert abs(curve[:,1].min()-m['validation_mae'])<1e-8
    assert int(curve[curve[:,1].argmin(),0])==m['best_epoch']
    val=float(np.abs(saved[:ck['val_end']-ck['fit_end']]-d['y'][ck['fit_end']:ck['val_end']]).mean())
    assert abs(val-m['validation_mae'])<1e-4
    return saved,m,meanerr


def main():
    setup(17);assert hashlib.sha256(Path('train_prices_15min.csv').read_bytes()).hexdigest()==SHA
    dates,y=read_targets('train_prices_15min.csv');d=np.load(ROOT/'data.npz')
    np.testing.assert_array_equal(y,d['y']);np.testing.assert_array_equal(dates,d['dates'])
    np.testing.assert_array_equal(y.mean(-1),d['h']);np.testing.assert_array_equal(baseline(y.mean(-1)),d['b'])
    q,r=features(dates,d['h'],dates,y);np.testing.assert_array_equal(q,d['q']);np.testing.assert_array_equal(r,d['r'])
    # Independently check every history block's source date, signed error and mask.
    lookup={str(date):yy for date,yy in zip(dates,y)}
    for i,date in enumerate(dates):
        for block,lag in enumerate(range(7,0,-1)):
            src=lookup.get(str((pd.Timestamp(date)-pd.Timedelta(days=lag)).date()))
            z=r[i,block*24:(block+1)*24,:40].reshape(24,4,10)
            if src is None:assert not z.any()
            else:
                np.testing.assert_array_equal(z[:,:,8],np.ones((24,4)))
                np.testing.assert_allclose(z[:,:,:4],((src-baseline(src.mean(-1)[None])[0])/20).transpose(1,0,2),atol=0,rtol=0)
    for idx in [120,302,333]:
        changed=y.copy();changed[idx:]+=1e5
        qq,rr=features(dates[idx:idx+1],d['h'][idx:idx+1],dates,changed)
        np.testing.assert_array_equal(qq,q[idx:idx+1]);np.testing.assert_array_equal(rr,r[idx:idx+1])
    s=json.loads((ROOT/'selection.json').read_text());assert [row['config'] for row in s['candidates']]==configs()
    report={};max_mean=0.;replays={}
    for row in s['candidates']:
        pred,m,err=replay(ROOT/'trials'/str(row['config']['id']),d);max_mean=max(max_mean,err);replays[row['config']['id']]=pred
    interpolation_error=np.abs(d['b'][333:]-y[333:]);fixed_worst=np.argsort(interpolation_error.mean((1,2,3)))[-3:]
    for family,row in s['by_family'].items():
        expected=min([rr for rr in s['candidates'] if rr['config']['family']==family],key=lambda rr:rr['validation_mae'])
        assert row==expected
        err=np.abs(replays[row['config']['id']][31:]-y[333:]);daily=err.mean((1,2,3))
        report[family]=dict(**row,september_mae=float(err.mean()),by_zone=err.mean((0,2,3)).tolist(),daily_mae=daily.tolist(),fixed_interpolation_worst3_mae=float(daily[fixed_worst].mean()))
    assert s['overall']==min(s['candidates'],key=lambda rr:rr['validation_mae'])
    rolling=json.loads((ROOT/'rolling.json').read_text());assert len(rolling['candidates'])==20
    roll_errors={f:[] for f in report};roll={}
    for row in rolling['candidates']:
        family=row['config']['family'];cfg=row['config']
        assert cfg['width']==32 and cfg['lr']==.001 and cfg['seed']==17
        assert row['fit_end']==row['val_end']-15 and (row['val_end'],row['stop']) in [(181,212),(212,243),(243,273),(273,302)]
        pred,m,err=replay(Path(row['folder']),d);max_mean=max(max_mean,err)
        errors=np.abs(pred[15:]-y[row['val_end']:row['stop']]);roll_errors[family].append(errors)
    for f,errs in roll_errors.items():
        e=np.concatenate(errs);assert len(e)==121;roll[f]=dict(mae=float(e.mean()),fold_mae=[float(x.mean()) for x in errs],by_zone=e.mean((0,2,3)).tolist())
    out=dict(by_family=report,rolling=roll,interpolation_september_mae=float(interpolation_error.mean()),selection_id=s['overall']['config']['id'],caveat='September developmental; earlier fixed-config folds also historical development, not a new blind test.')
    write(ROOT/'metrics.json',out)
    text=['# S4D/S5 history pilot — audited results','',out['caveat'],'','| Family | Parameters | August MAE | September MAE | Earlier rolling MAE |','|---|---:|---:|---:|---:|']
    for f,row in report.items():text.append(f"| {f} | {row['parameters']} | {row['validation_mae']:.6f} | {row['september_mae']:.6f} | {roll[f]['mae']:.6f} |")
    text+=['',f"Overall August-selected trial: {out['selection_id']}. Interpolation September MAE: {out['interpolation_september_mae']:.6f}.",'','Earlier rolling models use fixed width32/lr.001/seed17, not August-selected hyperparameters. All models share current compact causal summaries; compact control omits only the sequential history encoder. TCN has more parameters; report counts rather than claim parameter parity.','',f'Maximum hourly mean error: {max_mean:.3g}. All56 checkpoints replayed. No deployment changed.']
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    write(ROOT/'audit.json',dict(passed=True,checkpoints=56,source_hash=True,causal_features=True,train_only_normalization=True,selection_august_only=True,history_calendar_verified=True,max_hourly_mean_error=max_mean))
    print('AUDIT PASSED',out,flush=True)


if __name__=='__main__':main()
