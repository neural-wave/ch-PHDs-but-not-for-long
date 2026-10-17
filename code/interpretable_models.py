"""Compact physical features and transparent mean-conserving linear/additive models."""
import numpy as np
import pandas as pd

# Orthonormal zero-mean basis: half-hour contrast, then two quarter contrasts.
BASIS=np.array([[.5,.5,-.5,-.5],[1/np.sqrt(2),-1/np.sqrt(2),0,0],[0,0,1/np.sqrt(2),-1/np.sqrt(2)]])


def compact_features(dates,h,history_dates,history_y,history=False,interpolation_days=0):
    hist={str(d):yy-yy.mean(-1,keepdims=True) for d,yy in zip(history_dates,history_y)}
    names=['hourly_price','left_slope','right_slope','day_mean','day_std']+[f'zone_{z}_difference' for z in range(4)]+['hour_sin','hour_cos','hour_sin2','hour_cos2','weekday_sin','weekday_cos','year_sin','year_cos','weekend']
    if history:
        names += [f'{lag}_{contrast}' for lag in ['previous_day','previous_week','previous_7day_mean'] for contrast in ['half','quarter_first','quarter_second']]
    rows=[]
    for date,hh in zip(dates,h):
        dt=pd.Timestamp(str(date))
        priors=[]
        if history:
            for lag in [1,7]:
                priors.append(hist.get(str((dt-pd.Timedelta(days=lag)).date()),np.zeros((4,24,4)))@BASIS.T)
            before=[v for d,v in hist.items() if str((dt-pd.Timedelta(days=7)).date())<=d<str(date)]
            priors.append((np.mean(before,axis=0) if before else np.zeros((4,24,4)))@BASIS.T)
        day=[]
        for z in range(4):
            zone=[]
            for hour in range(24):
                row=[hh[z,hour]/100,(hh[z,hour]-hh[z,max(hour-1,0)])/20,(hh[z,min(hour+1,23)]-hh[z,hour])/20,
                     hh[z].mean()/100,hh[z].std()/100]
                row += ((hh[:,hour]-hh[z,hour])/20).tolist()
                row += [np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(4*np.pi*hour/24),np.cos(4*np.pi*hour/24),
                        np.sin(2*np.pi*dt.dayofweek/7),np.cos(2*np.pi*dt.dayofweek/7),np.sin(2*np.pi*dt.dayofyear/365.25),
                        np.cos(2*np.pi*dt.dayofyear/365.25),float(dt.dayofweek>=5)]
                if history:
                    row += np.concatenate([v[z,hour]/20 for v in priors]).tolist()
                zone.append(row)
            day.append(zone)
        rows.append(day)
    raw=np.asarray(rows,dtype=np.float64)
    if interpolation_days:
        from interpolation_history import interpolation_history
        extra,extra_names=interpolation_history(dates,history_dates,history_y,interpolation_days)
        raw=np.concatenate([raw,extra],axis=-1);names+=extra_names
    return raw,names


def design(raw,mu,sd,knots=None):
    xx=(raw-mu)/sd
    values=[np.ones(xx.shape[:-1]+(1,)),xx]
    if knots is not None:
        for knot in knots:
            values.append(np.maximum(xx-knot,0))
    return np.concatenate(values,axis=-1)


def infer(bundle,dates,h,history_dates,history_y):
    from reconstruction import baseline
    x,_=compact_features(dates,h,history_dates,history_y,bundle['history'],bundle.get('interpolation_days',0))
    coefficients=[]
    for z,state in enumerate(bundle['zones']):
        xx=design(x[:,z],state['mu'],state['sd'],state['knots'])
        coefficients.append(xx@state['beta'])
    c=np.stack(coefficients,axis=1)
    anchor=baseline(h) if bundle['anchor']=='interpolation' else np.broadcast_to(h[...,None],h.shape+(4,))
    pred=anchor+20*c@BASIS
    return pred
