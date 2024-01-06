import torch.nn as nn
import torch.nn.functional as F

from INet.training.loss.MMDLoss import MMDLoss


class LRSADTLMLoss(nn.Module):
    def __init__(self, source_output, source_label, source_feature,
                 target_feature, domain_output, domain_label, lamda, **kwargs):
        super(LRSADTLMLoss, self).__init__()
        self.source_output = source_output
        self.source_feature = source_feature
        self.target_feature = target_feature
        self.domain_output = domain_output
        self.domain_label = domain_label
        self.source_label = source_label
        self.lamda = lamda

    def forward(self, source_output, source_label, source_feature,
                target_feature, domain_output, domain_label, lamda):
        class_loss = F.cross_entropy(source_output, source_label)
        mmd_loss = MMDLoss(source_feature, target_feature)
        domain_loss = F.binary_cross_entropy(domain_output, domain_label)
        return mmd_loss + class_loss + lamda * domain_loss
