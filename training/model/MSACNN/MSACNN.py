from torch import nn
from einops.layers.torch import Rearrange
from torchsummary import summary


class MSACNN(nn.Module):

    def __init__(self):
        super(MSACNN, self).__init__()
        self.model1 = nn.Sequential(
            Rearrange('b e h -> b h e'),
            nn.Conv1d(3, 16, kernel_size=64, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 13),
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    summary(MSACNN(), (1024, 3), device="cpu")