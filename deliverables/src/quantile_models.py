"""Frozen champion components copied from trainq.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import feature_pipeline as p


TAUS=np.array([.025,.05,.10,.20,.25,.50,.75,.80,.90,.95,.975])


INTERVALS=[(.50,4,6),(.80,2,8),(.90,1,9),(.95,0,10)]


class Quantiles(nn.Module):
    def __init__(self,cfg,data):
        super().__init__();self.family=cfg['family'];self.two_d=self.family=='cnn2d_analog'
        if self.two_d:
            body=p.Global2D();self.ch,self.aux=body.channels,body.aux
            self.conv,self.fusion,self.full_curve=body.conv,body.fusion,body.full_curve
            self.head=nn.Conv1d(16,44,1);self.feature='cnn'
        else:
            family,self.feature=self.family.split('_',1);n=data[self.feature].shape[-1]
            self.register_buffer('knots',torch.tensor(data[self.feature+'_knots']))
            if family in ['linear','additive']:
                self.head=nn.Linear(n*(4 if family=='additive' else 1),43)
            elif family=='mlp':
                self.body=nn.Sequential(nn.Linear(n,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
                self.head=nn.Linear(32,43)
            else:
                self.body=nn.Sequential(nn.Conv1d(n,16,5,padding=2,padding_mode='replicate'),nn.SiLU(),
                                        nn.Conv1d(16,16,5,padding=2,padding_mode='replicate'),nn.SiLU())
                self.head=nn.Conv1d(16,43,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def initialize(self,target):
        axes=(0,2,3) if self.two_d else (0,1,2)
        empirical=np.quantile(target,TAUS,axis=axes).T
        below=np.concatenate([np.maximum(-empirical[:,4:5],.005),np.maximum(np.diff(empirical[:,:5],axis=1)[:,::-1],.005)],-1)
        above=np.concatenate([np.maximum(empirical[:,6:7],.005),np.maximum(np.diff(empirical[:,6:],axis=1),.005)],-1)
        gaps=np.concatenate([below,above],-1)
        with torch.no_grad():
            self.head.bias[4 if self.two_d else 3:].copy_(torch.as_tensor(np.log(np.expm1(gaps)).ravel(),dtype=self.head.bias.dtype))

    def forward(self,x,curve):
        if self.two_d:
            grid=x[:,:-self.aux].reshape(-1,self.ch,16,96)
            encoded=F.relu(self.conv(F.pad(grid,(3,4,0,0),mode='replicate'))).flatten(1,2)
            fused=F.relu(self.fusion(torch.cat([encoded,x[:,-self.aux:]],1))+self.full_curve(curve)[:,:,None])
            raw=self.head(fused)
            median=raw[:,:4].reshape(-1,4,24,4)
            increments=raw[:,4:].reshape(-1,4,10,96).permute(0,1,3,2).reshape(-1,4,24,4,10)
        else:
            if self.family.startswith('additive'):
                x=torch.cat([x]+[(x-k).relu() for k in self.knots],-1)
            if self.family.startswith('cnn'):
                b,z,t,n=x.shape
                raw=self.head(self.body(x.reshape(b*z,t,n).transpose(1,2))).transpose(1,2).reshape(b,z,t,43)
            else:
                raw=self.head(self.body(x) if hasattr(self,'body') else x)
            a,u,v=raw[...,:3].unbind(-1)
            median=torch.stack([a+u,a-u,-a+v,-a-v],-1)
            increments=raw[...,3:].reshape(*raw.shape[:-1],4,10)
        median=median-median.mean(-1,keepdim=True)
        inc=F.softplus(increments)+1e-6
        lower=(median[...,None]-inc[...,:5].cumsum(-1)).flip(-1)
        upper=median[...,None]+inc[...,5:].cumsum(-1)
        return torch.cat([lower,median[...,None],upper],-1)


def physical(raw,anchor):
    raw=np.asarray(raw,dtype='float64')
    raw=raw-raw[...,5].mean(-1,keepdims=True)[...,None]
    result=anchor[...,None]+20*raw
    assert np.isfinite(result).all() and np.diff(result,axis=-1).min()>=-1e-9
    return result


def intervals(cal_q,cal_y,test_q):
    """Widen-only finite-sample CQR, then deterministic nesting."""
    result={};previous_lower=previous_upper=None;previous_cal_lower=previous_cal_upper=None
    for coverage,lo,hi in INTERVALS:
        scores=np.maximum.reduce([cal_q[...,lo]-cal_y,cal_y-cal_q[...,hi],np.zeros_like(cal_y)])
        pooled=scores.transpose(0,2,1,3).reshape(-1,4,4);rank=math.ceil((len(pooled)+1)*coverage)
        assert rank<=len(pooled)
        correction=np.sort(pooled,axis=0)[rank-1][None,:,None,:]
        lower=test_q[...,lo]-correction;upper=test_q[...,hi]+correction
        cal_lower=cal_q[...,lo]-correction;cal_upper=cal_q[...,hi]+correction
        if previous_lower is not None:
            lower=np.minimum(lower,previous_lower);upper=np.maximum(upper,previous_upper)
            cal_lower=np.minimum(cal_lower,previous_cal_lower);cal_upper=np.maximum(cal_upper,previous_cal_upper)
        level=round(coverage*100)
        result[level]=dict(lower=lower,upper=upper,cal_lower=cal_lower,cal_upper=cal_upper,
                           correction=correction[0,:,0,:],rank=rank,n=len(pooled),coverage=coverage)
        previous_lower,previous_upper,previous_cal_lower,previous_cal_upper=lower,upper,cal_lower,cal_upper
    return result
