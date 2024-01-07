from torch import nn


class LightweightNet(nn.Module):
    def __init__(self):
        super(LightweightNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(16),
        )

        self.relu = nn.ReLU()

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )

    def forward(self, x):

        x = self.layer1(x)
        x1 = self.block1(x)
        x1 = x1 + self.res1(x)

        x2 = self.relu(x1)
        x2_1 = self.block2(x2)
        x2 = x2 + x2_1

        x3 = self.relu(x2)
        x3_1 = self.block3(x3)
        x3 = x3_1 + self.res2(x3)

        return self.fc(x3)