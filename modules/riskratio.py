import torch
from torch import nn


class _RiskRatio(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, z, s):
        p = s.mean()
        prob = s * p + self.tau * (1 - s) * (1 - p)
        _s = 2 * s - 1
        return torch.mean(torch.div(self.surrogate(z, _s), prob) - self.tau)

    def get_01score(self, z, s):
        y_pred = torch.heaviside(z, torch.tensor([0.0]))
        p1 = torch.mean(s * y_pred) / torch.mean(s)
        p2 = torch.mean((1 - s) * y_pred) / torch.mean(1 - s)
        if p2 == 0:
            p2 = 1e-4
        return p1 / p2


class LinearRiskRatio(_RiskRatio):
    def __init__(self, tau):
        super().__init__(tau)
        self.surrogate = lambda z, _s: z * _s


class LogisticRiskRatio(_RiskRatio):
    def __init__(self, tau):
        super().__init__(tau)
        self.surrogate = lambda z, _s: torch.log(1 + torch.exp(z * _s))


class HingeRiskRatio(_RiskRatio):
    def __init__(self, tau):
        super().__init__(tau)
        self.surrogate = lambda z, _s: torch.clamp(1 + z * _s, min=0)


class SquaredRiskRatio(_RiskRatio):
    def __init__(self, tau):
        super().__init__(tau)
        self.surrogate = lambda z, _s: torch.square(z * _s + 1)


class ExponentialRiskRatio(_RiskRatio):
    def __init__(self, tau):
        super().__init__(tau)
        self.surrogate = lambda z, _s: torch.exp(z * _s)
