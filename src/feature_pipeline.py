"""Frozen champion components copied from experiment.py.

Only required definitions are retained. See provenance/source_map.json.
Feature rules, layer definitions and tensor names are preserved.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from price_data import rich_features
from compact_features import compact_features
from analog_cnn import make_features, Small2D, dimensions


def weights_for(dates, y, b, mask, objective):
    """Compute weights from fitting dates only; mean1 changes just a common scale."""
    indices = np.flatnonzero(mask)
    months = np.asarray([str(d)[:7] for d in dates])[indices]
    errors = np.abs(y[indices] - b[indices]).mean((1, 2, 3))
    weights = np.ones(len(indices), dtype=np.float64)
    details = {}
    for month in sorted(set(months)):
        ids = months == month
        denom = float(errors[ids].mean())
        assert denom > 0
        details[month] = dict(days=int(ids.sum()), interpolation_mae=denom)
        if objective == 'ratio':
            weights[ids] = 1 / (int(ids.sum()) * denom)
    weights /= weights.mean()
    return weights, details


def features(dates, h, hd, hy):
    rich = rich_features(dates, h, hd, hy).reshape(len(dates), 4, 24, -1)
    cnn = make_features(dates, h, hd, hy, 'context')
    analog = cnn[:, -4:].reshape(len(dates), 4, 24, 4).astype('float64') * 20
    analog -= analog.mean(-1, keepdims=True)
    compact = compact_features(dates, h, hd, hy, True, interpolation_days=7)[0]
    curve = np.broadcast_to(h.reshape(len(dates), 1, 1, 96) / 100, (len(dates), 4, 24, 96))
    compact = np.concatenate([compact, analog / 20, curve], axis=-1)
    return dict(rich=rich.astype('float32'), compact=compact.astype('float32'),
                cnn=cnn, analog=analog, curve=(h.reshape(len(dates), 96) / 100).astype('float32'))


class Global2D(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = dict(width=16, clock=True, residual=True, analog=90, context=5)
        base = Small2D(cfg)
        self.channels, self.aux = dimensions(cfg)
        self.conv, self.fusion, self.output = base.conv, base.fusion, base.output
        self.full_curve = nn.Linear(96, 16)

    def forward(self, x, curve):
        grid = x[:, :-self.aux].reshape(-1, self.channels, 16, 96)
        encoded = F.relu(self.conv(F.pad(grid, (3, 4, 0, 0), mode='replicate'))).flatten(1, 2)
        fused = F.relu(self.fusion(torch.cat([encoded, x[:, -self.aux:]], 1)) + self.full_curve(curve)[:, :, None])
        out = self.output(fused).reshape(-1, 4, 24, 4)
        return out - out.mean(-1, keepdim=True)
