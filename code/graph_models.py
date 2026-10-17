"""Randomly initialized joint four-zone attention and temporal convolution."""
import torch
from torch import nn

class ZoneBlock(nn.Module):
    def __init__(self,width,dropout):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv1d(width,width,3,padding=1),nn.SiLU(),nn.Conv1d(width,width,3,padding=1))
        self.time_norm=nn.LayerNorm(width)
        self.attn=nn.MultiheadAttention(width,4,dropout=dropout,batch_first=True)
        self.node_norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.SiLU(),nn.Dropout(dropout),nn.Linear(width*2,width))
        self.ff_norm=nn.LayerNorm(width)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        # x is a day batch × four nodes × 24 hours × channels.
        batch,nodes,hours,width=x.shape
        t=self.temporal(x.reshape(batch*nodes,hours,width).transpose(1,2)).transpose(1,2).reshape_as(x)
        x=self.time_norm(x+self.drop(t))
        q=x.permute(0,2,1,3).reshape(batch*hours,nodes,width)
        message,_=self.attn(q,q,q,need_weights=False)
        q=self.node_norm(q+self.drop(message))
        x=q.reshape(batch,hours,nodes,width).permute(0,2,1,3)
        return self.ff_norm(x+self.drop(self.ff(x)))

class GraphNet(nn.Module):
    def __init__(self,n,width,blocks,dropout):
        super().__init__()
        self.input=nn.Linear(n,width)
        self.layers=nn.ModuleList([ZoneBlock(width,dropout) for _ in range(blocks)])
        self.output=nn.Linear(width,4)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        assert x.ndim==4 and x.shape[1:3]==(4,24)
        x=self.input(x)
        for layer in self.layers:x=layer(x)
        out=self.output(x)
        return out-out.mean(-1,keepdim=True)
