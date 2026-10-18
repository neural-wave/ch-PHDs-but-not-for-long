"""Frozen champion components copied from parallel_zone_cnn.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import datetime as dt
import numpy as np


AUX = 23


def build_inputs(dates, hourly, history_dates, history_y, history):
    hist={str(d):y for d,y in zip(history_dates,history_y)};out=[]
    for date,h in zip(dates,hourly):
        day=dt.date.fromisoformat(str(date));pairs=[];masks=[]
        for lag in (range(7,-1,-1) if history else [0]):
            key=str(day-dt.timedelta(days=lag))
            if lag:
                assert key<str(date)
                yy=hist.get(key);available=yy is not None
                yy=np.zeros((4,24,4)) if yy is None else yy
                hh=yy.mean(-1)
            else:
                # Query-day quarter prices are never read, even during training.
                yy=np.zeros((4,24,4));hh=h;available=False
            pairs.extend([yy.reshape(4,96)/100,np.repeat(hh,4,axis=-1)/100])
            masks.extend([np.full((1,96),float(available)),np.full((1,96),float(available or lag==0))])
        prices=np.stack(pairs,axis=1)  # zone, row, quarter; Q/H pairs for each day.
        mask=np.stack(masks,axis=1)
        grid=np.concatenate([prices,mask],axis=0).reshape(-1,96)
        left=h-np.pad(h,((0,0),(1,0)),mode='edge')[:,:24]
        right=np.pad(h,((0,0),(0,1)),mode='edge')[:,1:]-h
        aux=np.concatenate([np.repeat(h,4,axis=-1)/100,np.repeat(left,4,axis=-1)/100,
                            np.repeat(right,4,axis=-1)/100,np.eye(4)[np.arange(96)%4].T,
                            np.broadcast_to(np.eye(7)[day.weekday()][:,None],(7,96))])
        out.append(np.concatenate([grid,aux]))
    return np.asarray(out,dtype=np.float32)
