import torch
from torch import nn


class ZeroOneLoss(nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()
        self.surrogate = lambda z: torch.heaviside(-z, torch.tensor([0.0]))

    def forward(self, z, y):
        y_ = y * 2 - 1
        return torch.mean(self.surrogate(z * y_))

    def get_01score(self, z, y):
        y_hat = torch.heaviside(z, torch.tensor([0.0]))
        accuracy = torch.mean(torch.abs(y_hat - y))
        return accuracy
