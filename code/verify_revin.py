"""RevIN block: current-input statistics, finite gradients, isolated days and exact means."""
import torch
from revin_models import RevINGraph

torch.set_num_threads(2);torch.manual_seed(17)
x=torch.randn(2,4,24,5);h=torch.randn(2,4,24)*50-10
for enabled in [False,True]:
    net=RevINGraph(5,32,1,0,enabled).eval()
    torch.nn.init.normal_(net.net.output.weight,std=.03)
    both=net(x,h);single=net(x[:1],h[:1])
    assert both.shape==(2,4,24,4) and torch.isfinite(both).all()
    assert both.mean(-1).abs().max()<1e-5
    assert torch.allclose(both[:1],single,atol=1e-5)
    changed=h.clone();changed[1]=1e4
    assert torch.allclose(both[:1],net(x,changed)[:1],atol=1e-5)
    flat=torch.ones_like(h)*-25
    assert torch.isfinite(net(x,flat)).all()
    both.abs().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
print('PASS control and RevIN: finite output/gradients, constant/negative inputs, zero means, day independence')
