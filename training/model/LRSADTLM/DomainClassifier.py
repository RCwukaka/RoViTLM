from torch import nn


class DomainClassifier(nn.Module):
    def __init__(self, in_channel=1024):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )

    def forward(self, x):
        return self.layer(x)