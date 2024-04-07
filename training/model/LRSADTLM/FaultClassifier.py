from torch import nn


class FaultClassifier(nn.Module):
    def __init__(self, in_channel=1024, out_channel=44):
        super(FaultClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, out_channel),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        return self.layer(x)