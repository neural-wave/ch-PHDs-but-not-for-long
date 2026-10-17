"""Small PyTorch S4D/S5 cores; equations follow the cited official repositories.
S4D: state-spaces/s4/models/s4/s4d.py (Apache-2.0).
S5: lindermanlab/S5/s5/{ssm,ssm_init}.py (MIT).
This is a task-specific reimplementation, not an official benchmark reproduction.
"""
import math
import numpy as np
import torch
from torch import nn


def scan(a, b):
    """Inclusive affine associative scan, time axis 1; autograd safe."""
    aa=a[None,None,:].expand_as(b)
    out=b
    step=1
    while step<b.shape[1]:
        out=torch.cat([out[:,:step],out[:,step:]+aa[:,step:]*out[:,:-step]],1)
        aa=torch.cat([aa[:,:step],aa[:,step:]*aa[:,:-step]],1)
        step*=2
    return out


class S4D(nn.Module):
    def __init__(self,width,state):
        super().__init__();p=state//2
        self.log_decay=nn.Parameter(torch.full((width,p),math.log(.5)))
        self.freq=nn.Parameter(math.pi*torch.arange(p).float().repeat(width,1))
        self.log_dt=nn.Parameter(torch.empty(width).uniform_(math.log(.001),math.log(.1)))
        self.C=nn.Parameter(torch.randn(width,p,2)/math.sqrt(p))
        self.D=nn.Parameter(torch.ones(width))
    def discrete(self):
        a=torch.complex(-self.log_decay.exp(),self.freq);dt=self.log_dt.exp()[:,None]
        return torch.exp(a*dt),torch.expm1(a*dt)/a
    def kernel(self,length):
        a=torch.complex(-self.log_decay.exp(),self.freq);dt=self.log_dt.exp()[:,None]
        t=torch.arange(length,device=a.device)
        return 2*(torch.view_as_complex(self.C.contiguous())[:,:,None]*(torch.expm1(a*dt)/a)[:,:,None]*torch.exp((a*dt)[:,:,None]*t)).sum(1).real
    def forward(self,x):
        n=x.shape[1];u=x.transpose(1,2);k=self.kernel(n)
        out=torch.fft.irfft(torch.fft.rfft(u,n=2*n)*torch.fft.rfft(k,n=2*n)[None],n=2*n)[...,:n]
        return out.transpose(1,2)+x*self.D
    def reference(self,x):
        a,b=self.discrete();state=torch.zeros(x.shape[0],*a.shape,device=x.device,dtype=a.dtype);ys=[]
        for u in x.unbind(1):
            state=a*state+b*u[:,:,None]
            ys.append(2*(state*torch.view_as_complex(self.C.contiguous())).sum(-1).real+u*self.D)
        return torch.stack(ys,1)


class S5(nn.Module):
    def __init__(self,width,state):
        super().__init__();p=state//2
        # Normal part of HiPPO-LegS; use one conjugate half of its eigenbasis.
        v=np.sqrt(1+2*np.arange(state));hippo=-(np.tril(v[:,None]*v[None,:])-np.diag(np.arange(state)))
        low=np.sqrt(np.arange(state)+.5);normal=hippo+low[:,None]*low[None,:]
        freq,V=np.linalg.eigh(-1j*normal);V=V[:,:p]
        self.log_decay=nn.Parameter(torch.full((p,),math.log(.5)))
        self.freq=nn.Parameter(torch.tensor(freq[:p],dtype=torch.float32))
        self.log_dt=nn.Parameter(torch.empty(p).uniform_(math.log(.001),math.log(.1)))
        B=V.conj().T@(np.random.randn(state,width)/math.sqrt(state))
        self.B=nn.Parameter(torch.view_as_real(torch.tensor(B,dtype=torch.complex64)))
        self.C=nn.Parameter(torch.randn(width,p,2)/math.sqrt(2*p))
        self.D=nn.Parameter(torch.ones(width))
    def discrete(self):
        a=torch.complex(-self.log_decay.exp(),self.freq);dt=self.log_dt.exp()
        return torch.exp(a*dt),(torch.expm1(a*dt)/a)[:,None]*torch.view_as_complex(self.B.contiguous())
    def forward(self,x):
        a,b=self.discrete();bu=x.to(b.dtype)@b.T
        states=scan(a,bu)
        return 2*(states@torch.view_as_complex(self.C.contiguous()).T).real+x*self.D
    def reference(self,x):
        a,b=self.discrete();state=torch.zeros(x.shape[0],len(a),device=x.device,dtype=a.dtype);ys=[]
        for u in x.unbind(1):
            state=a*state+u.to(b.dtype)@b.T
            ys.append(2*(state@torch.view_as_complex(self.C.contiguous()).T).real+u*self.D)
        return torch.stack(ys,1)


class Block(nn.Module):
    def __init__(self,family,width,state):
        super().__init__();self.family=family;self.norm=nn.LayerNorm(width)
        if family=='s4d':self.core=S4D(width,state)
        elif family=='s5':self.core=S5(width,state)
        elif family=='gru':self.core=nn.GRU(width,width,batch_first=True)
        else:
            # Nine causal layers, receptive field 191 hours (>168-hour history).
            self.core=nn.ModuleList([nn.Conv1d(width,width,3,dilation=d) for d in [1,2,4,8,16,32,16,8,8]])
        self.mix=nn.Sequential(nn.Linear(width,2*width),nn.GLU(-1),nn.Dropout(.1))
    def forward(self,x):
        y=self.norm(x)
        if self.family=='gru':y=self.core(y)[0]
        elif self.family=='tcn':
            y=y.transpose(1,2)
            for layer in self.core:y=y+torch.nn.functional.silu(layer(torch.nn.functional.pad(y,(2*layer.dilation[0],0))))
            y=y.transpose(1,2)
        else:y=self.core(y)
        return x+self.mix(torch.nn.functional.gelu(y))


class Model(nn.Module):
    def __init__(self,cfg,query_dim,history_dim):
        super().__init__();self.cfg=cfg;w=cfg['width'];self.summary=cfg['family']=='compact'
        if not self.summary:
            self.project=nn.Linear(history_dim,w);self.block=Block(cfg['family'],w,cfg['state'])
        self.decoder=nn.Sequential(nn.Conv1d(query_dim+(0 if self.summary else 2*w),w,3,padding=1),nn.SiLU(),nn.Conv1d(w,w,3,padding=1),nn.SiLU(),nn.Conv1d(w,12,1))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,q,h):
        if not self.summary:
            z=self.block(self.project(h))
            q=torch.cat([q,z[:,-24:],z.mean(1,keepdim=True).expand(-1,24,-1)],-1)
        c=self.decoder(q.transpose(1,2)).transpose(1,2).reshape(-1,24,4,3).permute(0,2,1,3)
        a,u,v=c.unbind(-1)
        return torch.stack([a+u,a-u,-a+v,-a-v],-1)
