"""Independent NumPy artifact checks for learned segment lengths."""
import json
from pathlib import Path
import numpy as np
from reconstruction import read_targets,baseline
from segmented_frequency import numpy_basis,seasonal_coordinates
root=Path('results/segmented_frequency_20261017')
dates,y=read_targets(Path('train_prices_15min.csv'));h=y.mean(-1);anchor=baseline(h)
manifest=json.loads((root/'manifest.json').read_text());metrics=json.loads((root/'metrics.json').read_text())
cal=seasonal_coordinates(dates);checked=[]
for mode in ['global8','noon','constant','seasonal']:
    d=np.load(root/mode/'outputs.npz');pred=d['predictions'];basis=d['basis'];coef=d['coefficients']
    meta=metrics[mode]['selected'];bb=meta['b0']+meta['bs']*cal[:,0]+meta['bc']*cal[:,1]
    assert np.allclose(bb,d['boundaries'],atol=1e-10)
    assert bb.min()>=8-1e-6 and bb.max()<=16+1e-6
    assert coef.shape==(61,4,8) and basis.shape==(61,8,96)
    if mode!='global8':assert np.allclose(numpy_basis(bb,manifest['labels'])[302:],basis,atol=1e-10)
    reconstructed=anchor[302:].reshape(61,4,96)+np.einsum('dzk,dkt->dzt',coef,basis)
    assert np.max(np.abs(pred-reconstructed))<1e-8
    assert np.max(np.abs(pred.reshape(61,4,24,4).mean(-1)-h[302:]))<1e-8
    oracle=np.load(root/mode/'oracle.npy')
    for sl,key in [(slice(0,31),'august'),(slice(31,None),'september')]:
        truth=y[302:].reshape(61,4,96)[sl]
        mae=np.abs(pred[sl]-truth).mean();oa=np.abs(oracle[sl]-truth).mean()
        assert abs(mae-metrics[mode][key]['mae'])<1e-8
        assert abs(oa-metrics[mode]['oracle_'+key]['mae'])<1e-8
        assert oa<=mae+1e-6 and oa<=np.abs(anchor[302:].reshape(61,4,96)[sl]-truth).mean()+1e-6
    checked.append(mode)
(root/'independent_audit.json').write_text(json.dumps(dict(passed=True,checked=checked,
    checks='eight coefficient shape; NumPy basis and sinusoidal boundary replay; hourly means; recomputed MAEs; L1 oracle bounds'),indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,4.3))
names=checked;xx=np.arange(4)
axs[0].bar(xx-.18,[metrics[n]['september']['mae'] for n in names],.36,label='Predicted coefficients')
axs[0].bar(xx+.18,[metrics[n]['oracle_september']['mae'] for n in names],.36,label='Oracle coefficients')
axs[0].axhline(metrics['interpolation']['september']['mae'],ls='--',color='black',label='Interpolation')
axs[0].set(xticks=xx,xticklabels=['Global 8','Noon 4+4','Tuned 4+4','Seasonal 4+4'],ylabel='September MAE (EUR/MWh)');axs[0].tick_params(axis='x',labelrotation=15);axs[0].legend(fontsize=8)
for n in ['noon','constant','seasonal']:
    meta=metrics[n]['selected'];bb=meta['b0']+meta['bs']*cal[:,0]+meta['bc']*cal[:,1]
    axs[1].plot(np.array(dates,dtype='datetime64[D]'),bb,label=n)
axs[1].set(ylabel='Split hour',title='Learned boundary across the year',ylim=(8,16));axs[1].legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
print('INDEPENDENT AUDIT PASSED',flush=True)
