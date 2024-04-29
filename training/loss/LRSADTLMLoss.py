import torch.nn as nn

from INet.training.loss.CORALLoss import CORALLoss
from INet.training.loss.DomainLoss import DomainLoss
from INet.training.loss.FaultLoss import FaultLoss
from INet.training.loss.JMKMMD import JMKMMD
from INet.training.loss.MKMMD import MKMMD
from INet.training.loss.MMD import MMD


class LRSADTLMLoss(nn.Module):
    def __init__(self, type):
        super(LRSADTLMLoss, self).__init__()
        self.faultLoss = FaultLoss()
        self.domainLoss = DomainLoss()
        self.coralLoss = CORALLoss()
        self.JMKMMD = JMKMMD()
        self.MKMMD = MKMMD()
        self.MMD = MMD()
        self.type = type

    def forward(self, source_output, source_label, source_feature, target_output,
                target_feature, source_domain_output, target_domain_output, lamda, mu):
        class_loss = self.faultLoss(source_output, source_label)
        domain_loss = self.domainLoss(source_domain_output, target_domain_output)
        if self.type == 0:
            mkmmd = self.MKMMD(source_feature, target_feature)
            return class_loss + lamda * domain_loss + mu * mkmmd
        if self.type == 1:
            coralLoss = self.coralLoss(source_feature, target_feature)
            return class_loss + lamda * domain_loss + mu * coralLoss
        if self.type == 2:
            jmkmmd = self.JMKMMD(source_feature, target_feature)
            return class_loss + lamda * domain_loss + mu * jmkmmd
        if self.type == 3:
            mmd = self.MMD(source_feature, target_feature)
            return class_loss + lamda * domain_loss + mu * mmd

