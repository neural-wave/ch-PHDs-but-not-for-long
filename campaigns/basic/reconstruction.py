"""Shared, causal features and constrained models; no network access."""
import numpy as np
import pandas as pd
import torch
from torch import nn

ZONES = ['ZONE_0', 'ZONE_1', 'ZONE_2', 'ZONE_3']

def read_targets(path):
    df = pd.read_csv(path)
    keys = ['delivery_date', 'market_id', 'hour_index', 'quarter_index']
    assert not df.duplicated(keys).any()
    p = df.pivot(index='delivery_date', columns=keys[1:], values='price_eur_mwh')
    cols = pd.MultiIndex.from_product([ZONES, range(1,25), range(1,5)])
    p = p.reindex(columns=cols).sort_index()
    y = p.to_numpy().reshape(-1,4,24,4)
    assert np.isfinite(y).all()
    return p.index.to_numpy(), y

def baseline(h):
    out = np.empty(h.shape + (4,))
    for i in np.ndindex(h.shape[:-1]):
        out[i] = np.interp(np.arange(96)/4+.125, np.arange(24)+.5, h[i]).reshape(24,4)
    return out-out.mean(-1,keepdims=True)+h[...,None]

def features(dates, hourly, history_dates, history_y):
    """History is filtered by calendar date, including in validation and inference."""
    hist = {str(d): y for d,y in zip(history_dates,history_y)}
    rows = []
    for date,h in zip(dates,hourly):
        dt = pd.Timestamp(date)
        prior = sorted(d for d in hist if d < str(date))
        lag = []
        for days in [1,2,7]:
            key = str((dt-pd.Timedelta(days=days)).date())
            yy = hist.get(key)
            lag.append(np.zeros((4,24,4)) if yy is None else yy-yy.mean(-1,keepdims=True))
        for window in [7,28]:
            valid = [d for d in prior if d >= str((dt-pd.Timedelta(days=window)).date())]
            yy = np.array([hist[d] for d in valid])
            lag.append((yy-yy.mean(-1,keepdims=True)).mean(0) if len(yy) else np.zeros((4,24,4)))
        for z in range(4):
            for hour in range(24):
                offsets = np.clip(hour+np.arange(-3,4),0,23)
                local = h[:,offsets]
                cal = [np.sin(2*np.pi*dt.dayofyear/365.25), np.cos(2*np.pi*dt.dayofyear/365.25),
                       np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)]
                row = np.concatenate([h.ravel()/100, (h-h[z,hour]).ravel()/100,
                    local.ravel()/100, (local-h[z,hour]).ravel()/100,
                    np.eye(4)[z], np.eye(24)[hour], np.eye(7)[dt.dayofweek],cal,
                    np.concatenate([v[z,hour] for v in lag])/20])
                rows.append(row)
    return np.asarray(rows,dtype=np.float32)

class Net(nn.Module):
    def __init__(self, n, width, depth, dropout):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(n,width),nn.SiLU(),nn.Dropout(dropout)]
            n = width
        layers += [nn.Linear(n,4)]
        self.layers = nn.Sequential(*layers)
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):
        y = self.layers(x)
        return y-y.mean(-1,keepdim=True)

def project(y):
    return y-y.mean(-1,keepdims=True)
