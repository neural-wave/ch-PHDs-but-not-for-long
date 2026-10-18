import json
import numpy as np
import torch
from tune_parallel_cnn import ROOT,ARMS,configs,make_features,Small2D
from reconstruction import read_targets,baseline

d=np.load(ROOT/'data.npz');dates,y=read_targets('train_prices_15min.csv')
np.testing.assert_array_equal(y,d['y']);np.testing.assert_array_equal(dates,d['dates']);np.testing.assert_array_equal(baseline(y.mean(-1)),d['b'])
report=json.loads((ROOT/'metrics.json').read_text());selection=json.loads((ROOT/'selection.json').read_text())
rows=report['candidates'];assert len(rows)==len(configs())==42 and len({r['config']['id'] for r in rows})==42
assert selection['selected']['config']['id']==min(rows,key=lambda r:r['validation_mae'])['config']['id']
audit={}
for arm in ARMS:
    r=selection['by_arm'][arm];assert r['config']['id']==min([a for a in rows if a['config']['arm']==arm],key=lambda a:a['validation_mae'])['config']['id']
    raw=make_features(d['dates'],d['h'],d['dates'],d['y'],arm);f=np.load(ROOT/f'{arm}.npz')
    np.testing.assert_array_equal(f['mu'],raw[:302].mean((0,2),keepdims=True));np.testing.assert_array_equal(f['sd'],raw[:302].std((0,2),keepdims=True).clip(.05))
    np.testing.assert_array_equal(f['x'],np.clip((raw-f['mu'])/f['sd'],-15,15))
    folder=ROOT/'trials'/str(r['config']['id']);pred=np.load(folder/'predictions.npy');state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    net=Small2D(r['config']).cuda().eval();net.load_state_dict(state['state'])
    with torch.no_grad():rr=net(torch.tensor(f['x'][302:],device='cuda')).cpu().numpy().astype('float64')
    replay=d['b'][302:]+20*(rr-rr.mean(-1,keepdims=True));np.testing.assert_allclose(pred,replay,atol=1e-7,rtol=0)
    val=float(np.abs(pred[:31]-y[302:333]).mean());test=float(np.abs(pred[31:]-y[333:]).mean())
    assert abs(val-r['validation_mae'])<1e-4 and abs(test-report['by_arm'][arm]['september_mae'])<1e-10
    meanerror=float(np.abs(pred.mean(-1)-d['h'][302:]).max());assert meanerror<1e-8 and np.isfinite(pred).all()
    audit[arm]=dict(validation_mae=val,september_mae=test,hourly_mean_error=meanerror,replay_max_error=float(np.abs(pred-replay).max()))
(ROOT/'audit.json').write_text(json.dumps(dict(passed=True,by_arm=audit,checks=['official input equivalence','42 fits','August choices','train-only statistics','feature reconstruction','metric reconstruction','conservation','checkpoint replay']),indent=2))
lines=['# Focused parallel 2D CNN improvements','',f"Overall August-selected arm: **{report['selected_arm']}**.",'',
       'September is developmental; no model selected using September.','', '| Arm | Parameters | August MAE | September MAE |','|---|---:|---:|---:|']
for arm,r in report['by_arm'].items():lines.append(f"| {arm} | {r['parameters']} | {audit[arm]['validation_mae']:.6f} | {audit[arm]['september_mae']:.6f} |")
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('AUDIT PASSED',audit,flush=True)

# Export original parallel-row kernels of the August-selected model for study.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=selection['selected'];folder=ROOT/'trials'/str(r['config']['id']);state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
net=Small2D(r['config']);net.load_state_dict(state['state']);w=net.conv.weight.detach().numpy()
for channel in ([0,5] if r['config']['residual'] else [0]):
    n=w.shape[0];cols=4 if n==16 else 2;fig,axes=plt.subplots(4,cols,figsize=(cols*3.2,7),layout='constrained');lim=max(float(np.abs(w[:,channel]).max()),1e-8)
    for i,ax in enumerate(axes.flat):
        im=ax.imshow(w[i,channel],cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set(title=f'Filter {i}',yticks=[0,1],yticklabels=['Quarter','Hourly'],xticks=[0,3,7],xticklabels=[-3,0,4])
    fig.suptitle(f"August-selected {r['config']['arm']}; zone0 {'raw parallel rows' if channel==0 else 'interpolation error / hourly difference'}; normalized-input weights")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6);fig.savefig(folder/f'kernels_channel_{channel}.png',dpi=140);plt.close(fig)
