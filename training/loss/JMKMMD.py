from typing import Optional, Sequence
import torch
import torch.nn as nn
from INet.training.loss.kernels import GaussianKernel

# https://arxiv.org/abs/1605.06636

class JMKMMD(nn.Module):

    def __init__(self, linear: Optional[bool] = True, thetas: Sequence[nn.Module] = None):
        super(JMKMMD, self).__init__()
        self.kernels = ((GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.)), (GaussianKernel(alpha=1.5), GaussianKernel(alpha=2.)))
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in self.kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s.size(0))
        self.index_matrix = self._update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        layer_features = torch.cat([z_s, z_t], dim=0)
        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_kernels, theta in zip(self.kernels, self.thetas):
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


    def _update_index_matrix(self, batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                             linear: Optional[bool] = True) -> torch.Tensor:
        if index_matrix is None or index_matrix.size(0) != batch_size * 2:
            index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_matrix[s1, s2] = 1. / float(batch_size)
                    index_matrix[t1, t2] = 1. / float(batch_size)
                    index_matrix[s1, t2] = -1. / float(batch_size)
                    index_matrix[s2, t1] = -1. / float(batch_size)
            else:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                            index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for j in range(batch_size):
                        index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                        index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
        return index_matrix