"""Check graph interaction, independent day batches and hourly constraints."""
import torch
from graph_models import GraphNet

torch.set_num_threads(4);torch.manual_seed(17)
net=GraphNet(12,32,1,0).eval()
torch.nn.init.normal_(net.output.weight,std=.03)
x=torch.randn(2,4,24,12)
with torch.no_grad():
    out=net(x)
    assert out.shape==(2,4,24,4) and torch.isfinite(out).all()
    assert out.sum(-1).abs().max()<1e-6
    changed=x.clone();changed[:,3,12]+=2
    linked=net(changed)
    assert (linked[:,0]-out[:,0]).abs().max()>1e-5,'No cross-node interaction detected'
    changed=x.clone();changed[1]+=3
    independent=net(changed)
    assert torch.allclose(out[0],independent[0],atol=1e-6),'Day batches interact'
    permutation=torch.tensor([2,0,3,1])
    shuffled=net(x[:,permutation])
    assert torch.allclose(shuffled,out[:,permutation],atol=1e-6),'Node permutation changed the shared graph operation'
print('PASS: four-node interaction, temporal output shape, day independence, node equivariance and zero-mean residuals')
