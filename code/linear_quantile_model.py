"""One linear multi-quantile model with a mean-conserving hierarchical median."""
import numpy as np
import torch
from torch import nn

QUANTILES=np.array([.025,.05,.1,.2,.25,.5,.75,.8,.9,.95,.975])


class LinearQuantiles(nn.Module):
    def __init__(self,n):
        super().__init__()
        # Three median contrasts + ten nonmedian quantiles for each of four quarters.
        self.linear=nn.Linear(n,43,dtype=torch.float64)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self,x):
        values=self.linear(x);a,u,v=values[...,:1],values[...,1:2],values[...,2:3]
        median=torch.cat([a+u,a-u,-a+v,-a-v],-1)
        other=values[...,3:].reshape(*values.shape[:-1],4,10)
        return torch.cat([other[...,:5],median[...,None],other[...,5:]],-1)


def joint_loss(pred,truth):
    tau=torch.as_tensor(QUANTILES,device=pred.device,dtype=pred.dtype)
    error=truth[...,None]-pred
    pinball=torch.maximum(tau*error,(tau-1)*error).mean()
    median=(pred[...,5]-truth).abs().mean()
    return median+pinball,median,pinball


def physical_quantiles(raw,anchor,hourly):
    raw=np.asarray(raw,dtype=np.float64)
    # Only rounding correction; ideal hierarchical median residual already sums to zero.
    raw=raw-raw[...,5].mean(-1,keepdims=True)[...,None]
    quantiles=anchor[...,None]+20*raw
    median=quantiles[...,5]
    assert np.max(np.abs(median.mean(-1)-hourly))<1e-8
    crossing=float((np.diff(quantiles,axis=-1)<-1e-10).mean())
    # Rearrange around the fixed median; report raw crossing and both pinball scores.
    ordered=quantiles.copy()
    ordered[...,:5]=np.sort(np.minimum(quantiles[...,:5],median[...,None]),axis=-1)
    ordered[...,6:]=np.sort(np.maximum(quantiles[...,6:],median[...,None]),axis=-1)
    assert np.isfinite(ordered).all() and np.diff(ordered,axis=-1).min()>=-1e-10
    return ordered,quantiles,crossing


def feature_names():
    names=[f'today_zone{z}_hour{hr:02d}_price' for z in range(4) for hr in range(1,25)]
    names += [f'today_zone{z}_hour{hr:02d}_minus_target_price' for z in range(4) for hr in range(1,25)]
    names += [f'local_zone{z}_offset{delta:+d}_price' for z in range(4) for delta in range(-3,4)]
    names += [f'local_zone{z}_offset{delta:+d}_minus_target_price' for z in range(4) for delta in range(-3,4)]
    names += [f'target_zone{z}' for z in range(4)]+[f'target_hour{hr:02d}' for hr in range(1,25)]+[f'weekday{d}' for d in range(7)]
    names += ['year_sin','year_cos','hour_sin','hour_cos']
    names += [f'{lag}_quarter{q}' for lag in ['shape_lag1','shape_lag2','shape_lag7','shape_mean7','shape_mean28'] for q in range(1,5)]
    slopes=[f'{slope}_zone{z}' for slope in ['left_slope','right_slope','slope_sum','slope_curvature'] for z in range(4)]
    names += slopes+['absolute_'+s for s in slopes]
    names += ['day_mean','day_std','day_min','day_max','day_q25','day_q75']
    names += [f'interpolation_error_{lag}_quarter{q}' for lag in ['lag1','lag2','lag7','lag14','median7','median14','median28','median56','analog90_nearest10'] for q in range(1,5)]
    names += ['hour_linear','weekday_linear','month_linear','weekend']+[f'current_zone{z}_minus_target_price' for z in range(4)]
    assert len(names)==389
    return names


def numpy_forward(x,weight,bias):
    values=np.asarray(x)@np.asarray(weight).T+np.asarray(bias)
    a,u,v=values[...,:1],values[...,1:2],values[...,2:3]
    median=np.concatenate([a+u,a-u,-a+v,-a-v],-1)
    other=values[...,3:].reshape(*values.shape[:-1],4,10)
    return np.concatenate([other[...,:5],median[...,None],other[...,5:]],-1)
