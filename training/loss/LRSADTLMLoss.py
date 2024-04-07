import torch.nn as nn

from INet.training.loss.MK_MMDLoss import MK_MMDLoss
from INet.training.loss.DomainLoss import DomainLoss
from INet.training.loss.FaultLoss import FaultLoss


class LRSADTLMLoss(nn.Module):
    def __init__(self):
        super(LRSADTLMLoss, self).__init__()
        self.faultLoss = FaultLoss()
        self.domainLoss = DomainLoss()
        self.mmdLoss = MK_MMDLoss()

    def forward(self, source_output, source_label, source_feature,
                target_feature, source_domain_output, target_domain_output, lamda):
        class_loss = self.faultLoss(source_output, source_label)
        mmd_loss = self.mmdLoss(source_feature, target_feature)
        domain_loss = self.domainLoss(source_domain_output, target_domain_output)
        return class_loss + lamda * domain_loss
