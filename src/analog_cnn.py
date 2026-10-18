"""Frozen champion components copied from tune_parallel_cnn.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import datetime as dt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from history_grid import build_inputs, AUX
from price_data import baseline


ARMS={
    'control':dict(width=8,clock=False,residual=False,analog=0,context=1),
    'width16':dict(width=16,clock=False,residual=False,analog=0,context=1),
    'clock':dict(width=16,clock=True,residual=False,analog=0,context=1),
    'residual':dict(width=16,clock=True,residual=True,analog=0,context=1),
    'analog7':dict(width=16,clock=True,residual=True,analog=7,context=1),
    'analog90':dict(width=16,clock=True,residual=True,analog=90,context=1),
    'context':dict(width=16,clock=True,residual=True,analog=90,context=5),
}


def make_features(dates,h,history_dates,history_y,arm):
    cfg=ARMS[arm];base=build_inputs(dates,h,history_dates,history_y,True)
    hist={str(d):yy for d,yy in zip(history_dates,history_y)}
    hist_h={d:yy.mean(-1) for d,yy in hist.items()}
    errors={d:yy-baseline(hist_h[d][None])[0] for d,yy in hist.items()}
    allx=[]
    for i,(date,hh) in enumerate(zip(dates,h)):
        day=dt.date.fromisoformat(str(date));grid=base[i,:-AUX].reshape(5,16,96);aux=base[i,-AUX:]
        if cfg['residual']:
            residual=[]
            for lag in range(7,-1,-1):
                key=str(day-dt.timedelta(days=lag));available=lag>0 and key in hist
                residual.extend([errors[key].reshape(4,96)/20 if available else np.zeros((4,96)),
                    np.repeat(hist_h[key]-hh,4,axis=-1)/100 if available else np.zeros((4,96))])
            grid=np.concatenate([grid,np.stack(residual,axis=1)],axis=0)
        if cfg['clock']:
            q=np.arange(96);year=day.timetuple().tm_yday
            clock=np.stack([np.sin(2*np.pi*q/96),np.cos(2*np.pi*q/96),
                            np.full(96,np.sin(2*np.pi*year/365.25)),np.full(96,np.cos(2*np.pi*year/365.25))])
            aux=np.concatenate([aux,clock])
        if cfg['analog']:
            earliest=str(day-dt.timedelta(days=cfg['analog']));prior=sorted(d for d in hist if earliest<=d<str(date))
            analog=np.zeros((4,24,4))
            if prior:
                ph=np.array([hist_h[d] for d in prior]);pe=np.array([errors[d] for d in prior])
                for z in range(4):
                    for hour in range(24):
                        ix=np.clip(hour+np.arange(-2,3),0,23)
                        shape=hh[z,ix]-hh[z,hour]
                        dist=np.mean((ph[:,z,ix]-ph[:,z,hour,None]-shape)**2,axis=1)+.1*(ph[:,z,hour]-hh[z,hour])**2
                        nearest=np.argsort(dist)[:10];weights=1/(25+dist[nearest]);weights/=weights.sum()
                        analog[z,hour]=(pe[nearest,z,hour]*weights[:,None]).sum(0)
            aux=np.concatenate([aux,analog.reshape(4,96)/20])
        allx.append(np.concatenate([grid.reshape(-1,96),aux]))
    return np.asarray(allx,dtype=np.float32)


def dimensions(cfg):
    return (9 if cfg['residual'] else 5),23+4*int(cfg['clock'])+4*int(bool(cfg['analog']))


class Small2D(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.channels,self.aux=dimensions(cfg);w=cfg['width'];k=cfg['context']
        self.conv=nn.Conv2d(self.channels,w,(2,8),stride=(2,1))
        self.fusion=nn.Conv1d(w*8+self.aux,w,k,padding=k//2,padding_mode='replicate')
        self.output=nn.Conv1d(w,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        grid=x[:,:-self.aux].reshape(-1,self.channels,16,96)
        encoded=F.relu(self.conv(F.pad(grid,(3,4,0,0),mode='replicate'))).flatten(1,2)
        y=self.output(F.relu(self.fusion(torch.cat([encoded,x[:,-self.aux:]],1)))).reshape(-1,4,24,4)
        return y-y.mean(-1,keepdim=True)
