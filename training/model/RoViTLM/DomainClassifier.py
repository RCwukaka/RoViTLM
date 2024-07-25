from torch import nn


class DomainClassifier(nn.Module):
    def __init__(self, in_channel=1024):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)