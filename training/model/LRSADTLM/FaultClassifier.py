from torch import nn


class FaultClassifier(nn.Module):
    def __init__(self, in_channel=1024, out_channel=44):
        super(FaultClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_channel),
        )

    def forward(self, x):
        return self.layer(x)