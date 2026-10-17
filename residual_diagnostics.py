"""Paired day-level diagnostics; these already-viewed holdouts do not select models."""
import json
from pathlib import Path
import numpy as np

names=['rich_search_gpu1','zone_search_gpu1','catboost_search_gpu1']
loss={};truth=None;baseline=None
for name in names:
    arr=np.load(Path('results',name,'heldout_predictions.npz'),allow_pickle=True)
    if truth is None:truth=arr['observed'];baseline=arr['baseline']
    assert np.array_equal(arr['observed'],truth)
    loss[name]=np.abs(arr['predicted']-truth).mean((1,2,3))
loss['interpolation']=np.abs(baseline-truth).mean((1,2,3))
rng=np.random.default_rng(20261016);n=len(truth);block=7;repeats=10000
starts=rng.integers(0,n-block+1,size=(repeats,(n+block-1)//block))
inds=(starts[:,:,None]+np.arange(block)).reshape(repeats,-1)[:,:n]
comparisons={}
for before,after in [('interpolation','catboost_search_gpu1'),('rich_search_gpu1','zone_search_gpu1'),('zone_search_gpu1','catboost_search_gpu1')]:
    diff=loss[before]-loss[after];sampled=diff[inds].mean(1)
    comparisons[before+' -> '+after]=dict(mean_daily_mae_reduction=float(diff.mean()),
        moving_7day_block_bootstrap_95pct_interval=np.quantile(sampled,[.025,.975]).tolist())
result=dict(days=n,bootstrap_replicates=repeats,block_days=block,
    caveat='Already-viewed development holdout, correlated days and one season; descriptive uncertainty only, no secret-test or model-selection claim.',
    comparisons=comparisons)
Path('results/paired_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
