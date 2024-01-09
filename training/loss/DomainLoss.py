import torch
from torch import nn


class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()

    def forward(self, source_domain_output, target_domain_output):
        source_loss = self.get_adversarial_result(source_domain_output, True)
        target_loss = self.get_adversarial_result(target_domain_output, False)
        return source_loss + target_loss

    def get_adversarial_result(self, x, source=True):
        if source:
            domain_label = torch.ones(len(x), 1)
        else:
            domain_label = torch.zeros(len(x), 1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss_adv = loss_fn(x, domain_label.float().to(x.device))
        return loss_adv