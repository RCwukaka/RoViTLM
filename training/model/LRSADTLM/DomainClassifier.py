from torch import nn


class DomainClassifier(nn.Module):
    def __init__(self, num_class=1024):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_class, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)