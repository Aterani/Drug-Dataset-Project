from sklearn.base import BaseEstimator
from torch import nn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedNet, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes)
        )
        # self.fc_seq = nn.Sequential(
        #     nn.Linear(11, 30),
        #     # nn.BatchNorm1d(30),
        #     nn.ReLU(),
        #     nn.Linear(30, 16),
        #     # nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     # nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 4),
        #     # nn.BatchNorm1d(4),
        #     nn.ReLU(),
        # )
        # self.head = nn.Linear(4, 1)

    def forward(self, x):
        x = x.float()
        out = self.fc_seq(x)
        return out


