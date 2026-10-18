"""Mean-conserving direct and hourly→half-hour→quarter neural predictors."""
import torch
from torch import nn
from graph_models import ZoneBlock


class HierarchicalNet(nn.Module):
    def __init__(self, n, width=64, hierarchical=True, blocks=1, dropout=.1):
        super().__init__()
        self.hierarchical = hierarchical
        self.input = nn.Linear(n, width)
        self.block = ZoneBlock(width, dropout)
        self.additional = nn.ModuleList([ZoneBlock(width, dropout) for _ in range(blocks - 1)])
        self.coarse = nn.Linear(width, 1)
        self.fine = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 2))
        self.direct = nn.Linear(width, 4)
        for layer in [self.coarse, self.fine[-1], self.direct]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        hidden = self.block(self.input(x))
        for block in self.additional:
            hidden = block(hidden)
        if not self.hierarchical:
            q = self.direct(hidden)
            return q - q.mean(-1, keepdim=True)
        a = self.coarse(hidden)
        halves = torch.cat([a, -a], -1)
        fine = self.fine(torch.cat([hidden, halves], -1))
        u, v = fine[..., :1], fine[..., 1:]
        return torch.cat([a + u, a - u, -a + v, -a - v], -1)


def conformal_quantile(scores, alpha, axis=0):
    """Finite-sample order statistic, including infinity when data are insufficient."""
    import numpy as np
    n = scores.shape[axis]
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        shape = list(scores.shape)
        del shape[axis]
        return np.full(shape, np.inf)
    return np.take(np.sort(scores, axis=axis), rank - 1, axis=axis)
