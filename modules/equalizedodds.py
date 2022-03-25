import torch
from torch import nn


class _EqualizeOdds(nn.Module):
    def __init__(self, cond_y=1):
        super(_EqualizeOdds, self).__init__()
        self.cond_y = cond_y

    def forward(self, z, s, y):
        mask = (y == self.cond_y)
        s = torch.masked_select(s, mask)
        z = torch.masked_select(z, mask)
        p = s.mean()
        prob = s * p + (1 - s) * (1 - p)
        _s = 2 * s - 1
        return torch.mean(torch.div(self.surrogate(z, _s), prob) - 1)

    def get_01score(self, z, s, y):
        mask = (y == self.cond_y)
        s = torch.masked_select(s, mask)
        z = torch.masked_select(z, mask)
        y_pred = torch.heaviside(z, torch.tensor([0.0]))
        p1 = torch.mean(s * y_pred) / torch.mean(s)
        p2 = torch.mean((1 - s) * y_pred) / torch.mean(1 - s)
        return p1 - p2


class ZeroOneEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: (1 + _s) / 2 * torch.heaviside(z, torch.tensor([0.0])) \
                                       + (1 - _s) / 2 * torch.heaviside(-z, torch.tensor([1.0]))


class LinearEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: z * _s


class LogisticEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: torch.log(1 + torch.exp(z * _s))


class HingeEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: torch.clamp(1 + z * _s, min=0)


class SquaredEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: torch.square(z * _s + 1)


class ExponentialEqualizeOdds(_EqualizeOdds):
    def __init__(self, cond_y):
        super().__init__(cond_y)
        self.surrogate = lambda z, _s: torch.exp(z * _s)
