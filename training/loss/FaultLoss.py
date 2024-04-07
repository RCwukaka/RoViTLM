import torch
import torch.nn.functional as F
from torch import nn


class FaultLoss(nn.Module):
    def __init__(self):
        super(FaultLoss, self).__init__()

    def forward(self, source_output, source_label):
        return F.cross_entropy(source_output, source_label)
