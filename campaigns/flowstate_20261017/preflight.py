"""Short actual-GPU/model/input check before the unattended campaign."""
import importlib.metadata as metadata
import numpy as np
import torch
from torch import nn
from experiment import (ROOT, CHANNELS, configure, load_data, inputs, history_context,
                        ConditionalCNN, write)

configure()
dates, y, h = load_data()
i = 302
forecast = np.load(ROOT / 'forecasts' / f'{dates[i]}.npz')['forecast']
raw = inputs(dates[i], h[i], dates, y, forecast)
changed = y.copy(); changed[i:] += np.arange(4)*12345
np.testing.assert_array_equal(raw, inputs(dates[i], h[i], dates, changed, forecast))
np.testing.assert_array_equal(history_context(dates[i], dates, y), history_context(dates[i], dates, changed))
torch.manual_seed(17)
net = ConditionalCNN().cuda()
nn.init.normal_(net.output.weight, std=.01)
x = torch.tensor(raw[None], device='cuda', requires_grad=True)
point = net(x)
point[0, 0, 0, 0].backward()
assert x.grad[0, 1, 92].abs() > 0
assert x.grad[0, -8:].abs().sum() > 0
assert float(point.mean(-1).abs().max().detach()) < 1e-6
opt = torch.optim.AdamW(net.parameters(), lr=.001)
for _ in range(2):
    opt.zero_grad()
    pred = net(x.detach())
    loss = pred.abs().mean()
    loss.backward(); opt.step()
assert torch.isfinite(loss)
result = dict(passed=True, actual_flowstate_median_shape=list(forecast.shape),
              causal_context=True, causal_head=True, full_curve_gradient=True,
              flowstate_gradient=True, two_optimizer_steps=True,
              head_parameters=sum(p.numel() for p in net.parameters()),
              channels=CHANNELS, versions={p: metadata.version(p) for p in
                  ['torch','numpy','pandas','transformers','huggingface-hub','datasets','accelerate']})
write(ROOT / 'preflight.json', result)
print(result, flush=True)
