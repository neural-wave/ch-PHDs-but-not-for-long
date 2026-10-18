"""Small remote checks: mean preservation, ordering, loss and coefficient replay."""
import numpy as np
import torch
from linear_quantile_model import LinearQuantiles,QUANTILES,joint_loss,physical_quantiles,feature_names,numpy_forward
assert torch.cuda.is_available() and len(feature_names())==389 and len(QUANTILES)==11 and QUANTILES[5]==.5
torch.set_num_threads(4);torch.manual_seed(31)
net=LinearQuantiles(6).cuda()
with torch.no_grad():
    net.linear.weight.uniform_(-.2,.2);net.linear.bias.uniform_(-.2,.2)
x=torch.randn(3,6,device='cuda',dtype=torch.float64);raw=net(x);truth=torch.randn(3,4,device='cuda',dtype=torch.float64)
assert raw.shape==(3,4,11) and raw[...,5].mean(-1).abs().max()<1e-12
loss,mae,pb=joint_loss(raw,truth);assert torch.allclose(loss,mae+pb);loss.backward();assert torch.isfinite(net.linear.weight.grad).all()
with torch.no_grad():a=raw.cpu().numpy()
replay=numpy_forward(x.cpu().numpy(),net.linear.weight.detach().cpu().numpy(),net.linear.bias.detach().cpu().numpy())
assert np.max(np.abs(a-replay))<1e-12
anchor=np.ones((3,4))*100
ordered,_,crossing=physical_quantiles(a,anchor,np.ones(3)*100)
assert np.diff(ordered,axis=-1).min()>=0 and np.max(np.abs(ordered[...,5].mean(-1)-100))<1e-12
assert crossing>0 # deliberately unordered raw linear outputs exercise correction.
print('PASS:11 levels, joint loss/finite gradients, hierarchical means, ordering, independent coefficient replay',flush=True)
