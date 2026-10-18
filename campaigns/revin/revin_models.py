"""Input-derived reversible day/zone normalization for constrained residuals."""
import torch
from torch import nn
from graph_models import GraphNet

class RevINGraph(nn.Module):
    def __init__(self,n,width,blocks,dropout,revin=True):
        super().__init__()
        self.revin=revin
        self.affine_log_scale=nn.Parameter(torch.zeros(1,4,1))
        self.affine_bias=nn.Parameter(torch.zeros(1,4,1))
        self.net=GraphNet(n+7,width,blocks,dropout)
    def forward(self,x,h):
        # Only the supplied 24 hourly inputs define normalization statistics.
        mu=h.mean(-1,keepdim=True).detach()
        sd=torch.sqrt(h.var(-1,keepdim=True,unbiased=False)+1.).detach()
        gamma=self.affine_log_scale.clamp(-3,3).exp()
        z=(h-mu)/sd if self.revin else h/100
        z=z*gamma+self.affine_bias
        all_z=z.permute(0,2,1)[:,None].expand(-1,4,-1,-1)
        extra=torch.cat([z[...,None],all_z,mu[...,None].expand(-1,-1,24,-1)/100,
                         sd[...,None].expand(-1,-1,24,-1)/100],-1)
        raw=self.net(torch.cat([x,extra],-1))
        # The head predicts deviations: mean restoration is supplied by the baseline.
        if self.revin:raw=raw*sd[...,None]/(20*gamma[...,None])
        return raw-raw.mean(-1,keepdim=True)
