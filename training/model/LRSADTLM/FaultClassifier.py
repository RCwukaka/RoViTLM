from torch import nn


class FaultClassifier(nn.Module):
    def __init__(self, in_channel=1024, out_channel=44):
        super(FaultClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, 500),
            nn.ReLU(),
            nn.Linear(500, out_channel),
        )

    def forward(self, x):
        return self.layer(x)