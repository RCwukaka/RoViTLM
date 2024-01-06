from torch import nn


class DomainClassifier(nn.Module):
    def __init__(self, num_class = 256):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_class, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layer(x)