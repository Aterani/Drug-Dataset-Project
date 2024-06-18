from sklearn.base import BaseEstimator
from torch import nn


class ObesNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ObesNet, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 350),
            nn.ReLU(),
            nn.Linear(350, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = x.float()
        out = self.fc_seq(x)
        return out


