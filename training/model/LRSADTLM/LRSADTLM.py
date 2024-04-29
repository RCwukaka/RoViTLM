import torch
from einops import rearrange
from torch import nn
from torch.autograd import Function

from INet.training.model.LRSADTLM.DomainClassifier import DomainClassifier
from INet.training.model.LRSADTLM.FaultClassifier import FaultClassifier
from INet.training.model.LRSADTLM.LightweightNet import LightweightNet
from INet.training.model.ViT.ViT import ViT


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LRSADTLM(nn.Module):
    def __init__(self, net1=LightweightNet(), net2=ViT(), num_class=11):
        super(LRSADTLM, self).__init__()
        self.layerResNet = net1
        self.SAMNet = net2
        self.grl = ReverseLayerF()
        self.class_classifier = FaultClassifier(out_channel=num_class)
        self.domain_classifier = DomainClassifier()
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
        )

    def forward(self, source_x1, source_x2):
        source_x1 = rearrange(source_x1, "b (c w) h -> b c w h", c=1)
        source_feature1 = self.layerResNet(source_x1)
        source_x2 = rearrange(source_x2, "b (c w) h -> b c w h", c=1)
        source_feature2 = self.SAMNet(source_x2)
        source_feature = torch.cat((source_feature1, source_feature2), dim=1)
        source_feature = self.fc(source_feature)
        source_output = self.class_classifier(source_feature)

        reverse_source_feature = self.grl.apply(source_feature, 1.0)
        source_domain_output = self.domain_classifier(reverse_source_feature)

        return source_output, source_feature, source_domain_output
