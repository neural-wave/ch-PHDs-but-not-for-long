"""Grouped whole-day permutation reliance of frozen August-selected models."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from simple_search import model

GROUPS={
 'daily_hourly_curves':list(range(192)),
 'local_hourly_curves':list(range(192,248)),
 'zone_identity':list(range(248,252)),
 'hour_identity':list(range(252,276))+[285,286],
 'calendar':list(range(276,285))+list(range(381,385)),
 'historical_flat_residuals':list(range(287,307)),
 'slopes_and_curvature':list(range(307,339)),
 'daily_price_statistics':list(range(339,345)),
 'historical_interpolation_errors':list(range(345,381)),
 'cross_zone_spreads':list(range(385,389)),
}
# Overlapping diagnostic groups are deliberately reported separately.
GROUPS['all_history']=list(range(287,307))+list(range(345,381))
for i,label in enumerate(['lag1','lag2','lag7','lag14','median7','median14','median28','median56','analogs90']):
    GROUPS['interpolation_'+label]=list(range(345+4*i,349+4*i))
GROUPS['interpolation_lags1_2_7']=list(range(345,357))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    partition=[c for k,v in list(GROUPS.items())[:10] for c in v]
    assert sorted(partition)==list(range(389)) and len(partition)==len(set(partition))
    d=np.load(a.root/'data.npz');f=np.load(a.root/'rich.npz')
    selections=json.loads((a.root/'selection.json').read_text())['by_family']
    report=dict(method='20 grouped whole-day permutations within each evaluation month; fixed model and anchor',
                uncertainty='95% circular 7-day block bootstrap of paired daily MAE changes, 2000 resamples; descriptive',
                limitations='Correlated/redundant groups and fixed interpolation anchor limit attribution. Reliance is not retraining benefit. Whole-day permutation cannot assess zone/hour identity, which is constant across days; within-month calendar variation is limited. September previously inspected.',
                groups=GROUPS,models={})
    rng=np.random.default_rng(20261017)
    for family,selected in selections.items():
        cfg=selected['config'];saved=torch.load(a.root/'trials'/str(cfg['id'])/'model.pt',map_location='cpu',weights_only=False)
        net=model(cfg,389,f['knots']).eval();net.load_state_dict(saved['state'])
        def predict(x,anchor):
            with torch.no_grad():r=net(torch.tensor(x)).numpy().astype('float64')
            return anchor+20*(r-r.mean(-1,keepdims=True))
        results={}
        for split,lo,hi in [('august',302,333),('september',333,363)]:
            x=f['x'][lo:hi];truth=d['y'][lo:hi]
            anchor=d['b'][lo:hi] if cfg['anchor']=='interpolation' else d['h'][lo:hi,...,None]
            base=predict(x,anchor);daily=np.abs(base-truth).mean((1,2,3))
            assert abs(daily.mean()-selected['validation_mae' if split=='august' else 'september_mae'])<2e-5
            cached=np.load(a.root/'trials'/str(cfg['id'])/'predictions.npy')[lo-302:hi-302]
            assert np.max(np.abs(base-cached))<1e-4
            perms=[rng.permutation(len(x)) for _ in range(20)]
            starts=rng.integers(0,len(x),size=(2000,int(np.ceil(len(x)/7))))
            boot=((starts[...,None]+np.arange(7))%len(x)).reshape(2000,-1)[:,:len(x)]
            rows=[]
            for name,cols in GROUPS.items():
                changes=[]
                for perm in perms:
                    altered=x.copy();altered[...,cols]=x[perm][...,cols]
                    pred=predict(altered,anchor)
                    changes.append(np.abs(pred-truth).mean((1,2,3))-daily)
                changes=np.array(changes);paired=changes.mean(0)
                interval=np.quantile(paired[boot].mean(1),[.025,.975])
                neutral=x.copy();neutral[...,cols]=0
                rows.append(dict(group=name,channels=len(cols),delta_mae=float(paired.mean()),
                                 relative_percent=float(100*paired.mean()/daily.mean()),
                                 block_interval=interval.tolist(),permutation_sd=float(changes.mean(1).std()),
                                 zero_standardized_delta=float(np.abs(predict(neutral,anchor)-truth).mean()-daily.mean()),
                                 daily_delta=paired.tolist()))
            results[split]=dict(baseline_mae=float(daily.mean()),ranking=sorted(rows,key=lambda r:-r['delta_mae']))
            print(family,split,[(r['group'],round(r['delta_mae'],4)) for r in results[split]['ranking'][:6]],flush=True)
        report['models'][family]=dict(config=cfg,results=results)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'importance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=[g for g in list(GROUPS)[:10] if g not in ['zone_identity','hour_identity']]+['all_history']
    fig,axes=plt.subplots(1,4,figsize=(17,6),sharey=True)
    for ax,(family,r) in zip(axes,report['models'].items()):
        lookup={v['group']:v for v in r['results']['september']['ranking']}
        vals=[lookup[g]['delta_mae'] for g in groups]
        ax.barh(np.arange(len(groups)),vals);ax.axvline(0,color='black',lw=.7)
        ax.set(title=family,xlabel='Increase in September MAE (EUR/MWh)')
    axes[0].set_yticks(np.arange(len(groups)),[g.replace('_',' ') for g in groups]);axes[0].invert_yaxis()
    fig.suptitle('Frozen model reliance: grouped whole-day permutation (20 repeats)')
    fig.tight_layout();fig.savefig(a.out/'importance.png',dpi=180);fig.savefig(a.out/'importance.svg');plt.close(fig)

if __name__=='__main__':main()
