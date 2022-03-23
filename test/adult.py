import random

import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

from modules.loss import ZeroOneLoss
from modules.riskdifference import *

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return torch.flatten(x)

    def predict(self, x):
        z = self.forward(x)
        y = (z > 0).float()
        return torch.flatten(y)


class AdultData(Dataset):
    def __init__(self):
        npz = self.df = np.load("../data/adult-balanced.npz")
        self.s = torch.from_numpy(npz["s"])
        self.x = torch.from_numpy(npz["x"])
        self.y = torch.from_numpy(npz["y"])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.s[index], self.x[index], self.y[index]


def neuralnet(dataloader):
    net = Net(87, 1)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    risk_criterion = nn.BCEWithLogitsLoss()

    zoloss = ZeroOneLoss()
    zord = ZeroOneRiskDiff()

    for lf, rd_criterion in zip(
        [0.25, 0.13, 0.13, 0.07, 0.13],
        [
            LogisticRiskDiff(),
            LinearRiskDiff(),
            HingeRiskDiff(),
            SquaredRiskDiff(),
            ExponentialRiskDiff(),
        ],
    ):
        print(rd_criterion.__class__.__name__)
        for epoch in range(20):
            for i_batch, (s, X, y) in enumerate(dataloader):
                z = net(X)
                risk = risk_criterion(z, y)
                rd = rd_criterion(z, s)
                #######
                obj = risk + lf * rd

                optimizer.zero_grad()
                obj.backward()
                optimizer.step()

                if (i_batch == 0) & (epoch > 15):
                    accuracy = 1 - zoloss(z, y)
                    score = zord(z, s)
                    print(
                        f"Epoch: {epoch:2d} \tobj: {obj.item():.6f} \tarruracy: {accuracy:.6f} \trd: {score:.6f}"
                    )


if __name__ == "__main__":
    adult_dataset = AdultData()
    train_size = int(len(adult_dataset) * 1)
    test_size = len(adult_dataset) - train_size
    train_set, valid_set = random_split(adult_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_set, batch_size=train_size // 10, shuffle=True, num_workers=0
    )

    neuralnet(train_dataloader)
