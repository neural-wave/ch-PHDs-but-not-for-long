"""Independent artifact audit for daily frequency experiment."""
import json
from pathlib import Path
import numpy as np
from reconstruction import read_targets, baseline

root=Path('results/four_frequency_20261017')
dates,y=read_targets(Path('train_prices_15min.csv'))
h=y.mean(-1); b=baseline(h)
m=json.loads((root/'metrics.json').read_text())
bases=np.load(root/'basis.npz')
checked=[]
for name in ['fourier4','fourier4_freephase','pca4','pca8','pca16']:
    pred=np.load(root/(name+'_predictions.npy')).reshape(61,4,24,4)
    oracle=np.load(root/(name+'_oracle.npy')).reshape(61,4,24,4)
    assert np.isfinite(pred).all() and np.isfinite(oracle).all()
    assert np.abs(pred.mean(-1)-h[302:]).max()<1e-8
    for sl,label in [(slice(0,31),'august'),(slice(31,None),'september')]:
        truth=y[302:][sl]
        ae=np.abs(pred[sl]-truth).mean()
        oa=np.abs(oracle[sl]-truth).mean()
        assert abs(ae-m[name][label]['mae'])<1e-8
        assert abs(oa-m[name]['oracle_'+label]['mae'])<1e-8
        assert oa<=ae+1e-6
        assert oa<=np.abs(b[302:][sl]-truth).mean()+1e-6
    coeff=(pred-b[302:]).reshape(-1,96) @ bases[name].T
    replay=coeff @ bases[name]
    assert np.max(np.abs(replay-(pred-b[302:]).reshape(-1,96)))<1e-7
    if name in ['fourier4','pca4']: assert len(bases[name])==4
    checked.append(name)
assert m['pca72']['oracle_september']['mae']<1e-7
(root/'independent_audit.json').write_text(json.dumps(dict(passed=True,checked=checked,
    checks='recomputed MAEs, hourly means, rank-four output budget, basis-span reconstruction, L1 oracle bounds, full-rank capacity'),indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['fourier4','fourier4_freephase','pca4','pca8','pca16']
xx=np.arange(len(names)); fig,ax=plt.subplots(figsize=(9,4.5))
ax.bar(xx-.18,[m[n]['september']['mae'] for n in names],.36,label='Predicted weights')
ax.bar(xx+.18,[m[n]['oracle_september']['mae'] for n in names],.36,label='Oracle weights (target-derived)')
ax.axhline(m['interpolation']['september']['mae'],color='black',ls='--',label='Interpolation')
ax.set(xticks=xx,xticklabels=['Fourier 4','Fourier 4\nfree phase','PCA 4','PCA 8','PCA 16'],ylabel='MAE (EUR/MWh)',title='September development: daily residual weights')
ax.legend(); fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
print('INDEPENDENT AUDIT PASSED',flush=True)
