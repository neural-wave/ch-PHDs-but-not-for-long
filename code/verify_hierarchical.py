"""Tiny checks on the remote GPU, including causal history and finite quantiles."""
import numpy as np
import torch
from hierarchical_models import HierarchicalNet, conformal_quantile
from reconstruction import rich_features
from hierarchical_features import lagged_features

torch.set_num_threads(4)
assert torch.cuda.is_available()
for hier,blocks in [(True,1),(False,1),(True,2),(False,2)]:
    net = HierarchicalNet(5, 32, hier,blocks,.05).cuda().eval()
    # Nonzero parameters test conservation away from zero initialization.
    with torch.no_grad():
        for param in net.parameters():
            param.uniform_(-.1, .1)
    x = torch.randn(2, 4, 24, 5, device='cuda')
    q = net(x)
    assert torch.isfinite(q).all() and q.mean(-1).abs().max() < 1e-6
    if hier:
        hidden = net.block(net.input(x))
        for block in net.additional:
            hidden = block(hidden)
        a = net.coarse(hidden)
        halves = q.reshape(2,4,24,2,2).mean(-1)
        assert torch.allclose(halves, torch.cat([a,-a],-1), atol=1e-6)
    q.abs().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
assert conformal_quantile(np.arange(16), .1) == 15
assert conformal_quantile(np.arange(16), .5) == 8
assert np.isinf(conformal_quantile(np.arange(16), .05))
dates = np.array(['2026-08-01','2026-08-02','2026-08-03'], dtype=object)
rng = np.random.default_rng(2)
y = rng.normal(size=(3,4,24,4))
h = y.mean(-1)
original = rich_features(dates[1:2], h[1:2], dates, y)
changed = y.copy()
changed[1:] += 10000
assert np.array_equal(original, rich_features(dates[1:2], h[1:2], dates, changed))
for count in [0,1,3,7,14,28]:
    left = lagged_features(dates[1:2],h[1:2],dates,y,count)
    right = lagged_features(dates[1:2],h[1:2],dates,changed,count)
    assert np.array_equal(left,right) and left.shape == (96,333+4*count)
    if count == 0:
        past_changed = y.copy()
        past_changed[0,0,0,0] += 1234
        assert np.array_equal(left,lagged_features(dates[1:2],h[1:2],dates,past_changed,0))
print('PASS: GPU kernels, hierarchy conservation, gradients, conformal ranks, causal features', flush=True)
