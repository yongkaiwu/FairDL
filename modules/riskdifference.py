import torch
from torch import nn


class _RiskDiff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, s):
        p = s.mean()
        prob = s * p + (1 - s) * (1 - p)
        _s = 2 * s - 1
        return torch.mean(torch.div(self.surrogate(z, _s), prob) - 1)

    def get_01score(self, z, s):
        y_pred = torch.heaviside(z, torch.tensor([0.0]))
        p1 = torch.mean(s * y_pred) / torch.mean(s)
        p2 = torch.mean((1 - s) * y_pred) / torch.mean(1 - s)
        return p1 - p2


class ZeroOneRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: (1 + _s) / 2 * torch.heaviside(z, torch.tensor([0.0])) \
                                       + (1 - _s) / 2 * torch.heaviside(-z, torch.tensor([1.0]))


class LinearRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: z * _s


class LogisticRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: torch.log(1 + torch.exp(z * _s))


class HingeRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: torch.clamp(1 + z * _s, min=0)


class SquaredRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: torch.square(z * _s + 1)


class ExponentialRiskDiff(_RiskDiff):
    def __init__(self):
        super().__init__()
        self.surrogate = lambda z, _s: torch.exp(z * _s)
