import torch
from torch import nn


class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()

    def forward(self, source_domain_output, target_domain_output):
        return torch.mean(1 / torch.log(source_domain_output) + 1 / torch.log(1 - target_domain_output))
