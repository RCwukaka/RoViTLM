import torch.nn as nn

class INet(nn.Module):
    def __init__(self, in_dim=1024*3, n_hidden_1=1024, n_hidden_2=248, out_dim=13):
        super(INet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out