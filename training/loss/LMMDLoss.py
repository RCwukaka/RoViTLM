import torch
import numpy as np
from torch import nn


class LMMDLoss(nn.Module):
    def __init__(self, num_class=12, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(LMMDLoss, self).__init__()
        self.num_class = num_class
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.max_iter = max_iter

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")

        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)

            kernels = self.guassian_kernel(source, target,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
            return loss

    def cal_weight(self, source_y, target_y):

        n_class = source_y.shape[1]
        source_y_sum = torch.sum(source_y, dim=0)
        source_y_normalize = source_y / source_y_sum
        source_y_normalize[torch.isnan(source_y_normalize)] = 0

        target_y_sum = torch.sum(target_y, dim=0)
        target_y_normalize = target_y / target_y_sum
        target_y_normalize[torch.isnan(target_y_normalize)] = 0

        # Find common classes between source and target
        source_y_value = torch.argmax(source_y, dim=1).unsqueeze(1)
        target_y_value = torch.argmax(target_y, dim=1).unsqueeze(1)
        common_classes = torch.sparse.to_dense(torch.sets.intersection(source_y_value, target_y_value))

        # Create mask for common classes
        mask_arr = torch.zeros(source_y.size(0), n_class)
        mask_arr[torch.arange(source_y.size(0)), common_classes] = 1

        # Apply mask to normalize labels
        source_y_normalize = source_y_normalize * mask_arr
        target_y_normalize = target_y_normalize * mask_arr

        # Compute weight matrices
        weight_ss = torch.matmul(source_y_normalize, source_y_normalize.t()) / n_class
        weight_tt = torch.matmul(target_y_normalize, target_y_normalize.t()) / n_class
        weight_st = torch.matmul(source_y_normalize, target_y_normalize.t()) / n_class

        return weight_ss, weight_tt, weight_st

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
