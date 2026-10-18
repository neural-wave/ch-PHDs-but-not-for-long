"""Frozen champion components copied from reconstruction.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import numpy as np
import pandas as pd


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


def rich_features(dates,hourly,history_dates,history_y):
    """Explicit slopes and historical *interpolation errors*, with fixed causal rules."""
    basic=features(dates,hourly,history_dates,history_y)
    hist={str(d):yy for d,yy in zip(history_dates,history_y)}
    hist_h={d:yy.mean(-1) for d,yy in hist.items()}
    hist_error={d:yy-baseline(hist_h[d][None])[0] for d,yy in hist.items()}
    extra=[]
    for date,h in zip(dates,hourly):
        dt=pd.Timestamp(date)
        before=sorted(d for d in hist if d<str(date))
        # Slopes in EUR/MWh per hour; no artificial next-day hourly inputs.
        padded=np.pad(h,((0,0),(1,1)),mode='edge')
        left=h-padded[:,:24]; right=padded[:,2:]-h
        shaped=[]
        for days in [1,2,7,14]:
            d=str((dt-pd.Timedelta(days=days)).date())
            shaped.append(hist_error.get(d,np.zeros((4,24,4))))
        for window in [7,14,28,56]:
            ds=[d for d in before if d>=str((dt-pd.Timedelta(days=window)).date())]
            shaped.append(np.median([hist_error[d] for d in ds],axis=0) if ds else np.zeros((4,24,4)))
        # Similar hourly curves retrieve historical residuals; fixed distance/weights,
        # no inference-time model fitting or hyperparameter changes.
        ds=[d for d in before if d>=str((dt-pd.Timedelta(days=90)).date())]
        analog=np.zeros((4,24,4))
        if ds:
            hh=np.array([hist_h[d] for d in ds])
            ee=np.array([hist_error[d] for d in ds])
            for z in range(4):
                for hour in range(24):
                    inds=np.clip(hour+np.arange(-2,3),0,23)
                    q=h[z,inds]-h[z,hour]
                    v=hh[:,z,inds]-hh[:,z,hour,None]
                    # Local curve shape plus price level distinguishes regimes.
                    dist=np.mean((v-q)**2,axis=1)+(hh[:,z,hour]-h[z,hour])**2*.1
                    nearest=np.argsort(dist)[:10]
                    weights=1/(25+dist[nearest]); weights/=weights.sum()
                    analog[z,hour]=(ee[nearest,z,hour]*weights[:,None]).sum(0)
        for z in range(4):
            for hour in range(24):
                slopes=np.concatenate([left[:,hour],right[:,hour],(left+right)[:,hour],(right-left)[:,hour]])/20
                scale=np.array([h[z].mean(),h[z].std(),h[z].min(),h[z].max(),
                    np.quantile(h[z],.25),np.quantile(h[z],.75)])/100
                errors=np.concatenate([v[z,hour] for v in shaped]+[analog[z,hour]])/20
                # Quantities are explicit so small trees need fewer interacting splits.
                extra.append(np.concatenate([slopes, np.abs(slopes), scale, errors,
                    [hour/23,dt.dayofweek/6,(dt.month-1)/11,float(dt.dayofweek>=5)],
                    (h[:,hour]-h[z,hour])/20]))
    return np.concatenate([basic,np.asarray(extra,dtype=np.float32)],axis=1)
