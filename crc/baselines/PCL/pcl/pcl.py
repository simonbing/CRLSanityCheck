"""PCL model"""


import torch
import torch.nn as nn
from subfunc.showdata import *


# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(torch.reshape(x, (*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:])), dim=2)
        return m


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, h_sizes, in_dim, latent_dim, ar_order=1, pool_size=2):
        """ Network model for gaussian distribution with scale-mean modulations
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_dim: number of dimension
             ar_order: model order of AR
             pool_size: pool size of max-out nonlinearity
         """
        super(Net, self).__init__()
        assert ar_order == 1  # this model is only for AR(1)
        # h
        h_sizes_aug = [in_dim] + h_sizes
        h = [nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k]*pool_size) for k in range(1, len(h_sizes_aug)-1)]
        h.append(nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1], bias=False))
        self.h = nn.ModuleList(h)
        self.bn = nn.BatchNorm1d(num_features=latent_dim)
        self.maxout = Maxout(pool_size)
        self.w = nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=2, groups=latent_dim)
        self.a = nn.Parameter(torch.ones([1]))
        self.m = nn.Parameter(torch.zeros([1]))
        self.latent_dim = latent_dim

        # initialize
        for k in range(len(self.h)):
            torch.nn.init.xavier_uniform_(self.h[k].weight)
        torch.nn.init.constant_(self.w.weight[:, :, 0], 1)
        torch.nn.init.constant_(self.w.weight[:, :, 1], -1)

    def forward(self, x):
        """ forward
         Args:
             x: input [batch, time(t:t-p), dim]
         """
        batch_size, _, num_dim = x.shape

        # h
        h = x.reshape([-1, num_dim])  # [batch * ar, dim]
        for k in range(len(self.h)):
            h = self.h[k](h)
            if k != len(self.h)-1:
                h = self.maxout(h)
        h = self.bn(h)
        h = h.reshape([batch_size, -1, self.latent_dim])  # [batch, ar, dim]

        # Build r(y) ----------------------------------------------
        #   sum_i |w1i*hi(y1) + w2i*hi(y2) + bi| + a * (hi(y1))^2 + m
        #         ----------------------------     ----------------
        #                       Q                        Qbar
        #   di and ki are fixed to 0 because they have indeterminacy with scale and bias of hi.
        #   [w1i, w2i, bi] are initialized by [1, -1, 0].
        q = torch.sum(torch.abs(self.w(torch.permute(h, (0, 2, 1)))), dim=[1, 2])
        qbar = self.a * torch.sum(h[:, 0, :] ** 2, dim=[1])  # [batch]
        logits = - q + qbar + self.m

        return logits, h
