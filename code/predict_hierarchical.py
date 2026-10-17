"""Offline point and interval inference for the frozen Vast campaign."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from hierarchical_models import HierarchicalNet
from reconstruction import ZONES, read_targets, rich_features, baseline
from hierarchical_features import lagged_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--hourly', required=True)
    p.add_argument('--history', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = Path(a.model_dir)
    report = json.loads((folder / 'metrics.json').read_text())
    frame = pd.read_csv(a.hourly)
    keys = ['delivery_date','market_id','hour_index']
    assert not frame.duplicated(keys).any()
    cols = pd.MultiIndex.from_product([ZONES, range(1,25)])
    pivot = frame.pivot(index='delivery_date', columns=keys[1:], values='hourly_price_eur_mwh').reindex(columns=cols).sort_index()
    dates = pivot.index.to_numpy()
    h = pivot.to_numpy().reshape(-1,4,24)
    assert np.isfinite(h).all() and len(frame) == len(dates)*96
    hd, hy = read_targets(a.history)
    selected = report['selected']
    if selected == 'interpolation':
        pred = baseline(h)
    else:
        checkpoint = torch.load(folder / (selected + '.pt'), weights_only=False, map_location='cpu')
        cfg = checkpoint['config']
        x = lagged_features(dates,h,hd,hy,cfg.get('lag_days',-1)).reshape(len(dates),4,24,-1)
        x = np.clip((x-checkpoint['mu'])/checkpoint['sd'],-15,15)
        net = HierarchicalNet(checkpoint['feature_count'],cfg['width'],cfg['hierarchical'],cfg.get('blocks',1),cfg.get('dropout',.1)).to(a.device).eval()
        net.load_state_dict(checkpoint['state'])
        with torch.no_grad():
            raw = net(torch.tensor(x,device=a.device)).cpu().numpy().astype(np.float64)
        anchor = baseline(h) if cfg.get('anchor') == 'interpolation' else h[...,None]
        pred = anchor + 20*(raw-raw.mean(-1,keepdims=True))
    rows = [(d,zone,hr+1,q+1,pred[i,z,hr,q]) for i,d in enumerate(dates) for z,zone in enumerate(ZONES) for hr in range(24) for q in range(4)]
    output = pd.DataFrame(rows,columns=keys+['quarter_index','predicted_price_eur_mwh'])
    # Bounds are supplied separately from the five-column challenge submission.
    intervals = output.copy()
    for item in report['conformal']:
        kind = item['kind']
        if kind == 'marginal':
            radius = np.asarray(item['radius']).reshape(1,4,1,4)
        elif kind == 'hour_joint':
            radius = np.asarray(item['radius']).reshape(1,4,1,1)
        else:
            radius = item['radius']
        label = kind + '_' + str(round(item['nominal']*100))
        intervals[label+'_lower'] = (pred-radius).ravel()
        intervals[label+'_upper'] = (pred+radius).ravel()
    assert np.isfinite(pred).all() and np.abs(pred.mean(-1)-h).max()<1e-4
    output.to_csv(a.output,index=False)
    intervals.to_csv(str(Path(a.output).with_suffix(''))+'_intervals.csv',index=False)


if __name__ == '__main__':
    main()
