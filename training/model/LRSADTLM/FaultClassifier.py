from torch import nn


class FaultClassifier(nn.Module):
    def __init__(self, in_channel=256, out_channel=39):
        super(FaultClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)