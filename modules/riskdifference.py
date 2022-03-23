import torch
from torch import nn


class _RiskDiff(nn.Module):
    def __init__(self):
        super(_RiskDiff, self).__init__()

    def forward(self, z, s):
        _s = 2 * s - 1
        p = _s.mean() / 2 + 0.5
        prob = (1 + _s) / 2 * p + (1 - _s) / 2 * (1 - p)
        return torch.mean(torch.div(self.surrogate(z, _s), prob) - 1)


class ZeroOneRiskDiff(_RiskDiff):
    def __init__(self):
        super(ZeroOneRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: (1 + _s) / 2 * torch.heaviside(z, torch.tensor([0.0])) \
                                       + (1 - _s) / 2 * torch.heaviside(-z, torch.tensor([1.0]))


class LinearRiskDiff(_RiskDiff):
    def __init__(self):
        super(LinearRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: z * _s


class LogisticRiskDiff(_RiskDiff):
    def __init__(self):
        super(LogisticRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: torch.log(1 + torch.exp(z * _s))


class HingeRiskDiff(_RiskDiff):
    def __init__(self):
        super(HingeRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: torch.clamp(1 + z * _s, min=0)


class SquaredRiskDiff(_RiskDiff):
    def __init__(self):
        super(SquaredRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: torch.square(z * _s + 1)


class ExponentialRiskDiff(_RiskDiff):
    def __init__(self):
        super(ExponentialRiskDiff, self).__init__()
        self.surrogate = lambda z, _s: torch.exp(z * _s)
