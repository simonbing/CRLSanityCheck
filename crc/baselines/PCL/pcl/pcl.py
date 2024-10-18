"""PCL model"""


import torch
import torch.nn as nn
from crc.baselines.PCL.subfunc.showdata import *


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
    def __init__(self, h_sizes, in_dim, latent_dim, image_data, ar_order=1,
                 pool_size=2, conv=False):
        """ Network model for gaussian distribution with scale-mean modulations
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_dim: number of dimension
             ar_order: model order of AR
             pool_size: pool size of max-out nonlinearity
         """
        super(Net, self).__init__()
        assert ar_order == 1  # this model is only for AR(1)
        self.image_data = image_data
        self.conv = conv

        if self.conv:
            NormLayer = lambda d: nn.GroupNorm(num_groups=8, num_channels=d)

            h_dim = 64
            n_conv_layers = 2

            conv_layers = [
                nn.Sequential(
                    nn.Conv2d(3 if i_layer == 0 else h_dim,
                              h_dim,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False),
                    NormLayer(h_dim),
                    nn.SiLU(),
                    nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1,
                              padding=1,
                              bias=False),
                    NormLayer(h_dim),
                    nn.SiLU()
                ) for i_layer in range(n_conv_layers)
            ]

            self.conv_encoder = nn.Sequential(
                *conv_layers,
                nn.Flatten(),
                nn.Linear(16 * 16 * h_dim, 16 * h_dim),
                nn.LayerNorm(16 * h_dim),
                nn.SiLU(),
                nn.Linear(16 * h_dim, latent_dim)
            )
        else:
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
        if not self.conv:
            for k in range(len(self.h)):
                torch.nn.init.xavier_uniform_(self.h[k].weight)
        torch.nn.init.constant_(self.w.weight[:, :, 0], 1)
        torch.nn.init.constant_(self.w.weight[:, :, 1], -1)

    def forward(self, x):
        """ forward
         Args:
             x: input [batch, time(t:t-p), dim]
         """
        batch_size = x.shape[0]
        num_dim = x.shape[-1]
        if self.image_data:
            if not self.conv:
                x = torch.flatten(x, start_dim=2)

        if self.conv:
            h = torch.cat((x[:, 0, ...], x[:, 1, ...]))
            h = self.conv_encoder(h)
        else:
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

    def get_z(self, x_list):
        x = x_list[:, 0, ...]
        x_perm = x_list[:, 1, ...]

        x_cat = torch.cat((x, x_perm))

        _, h = self.forward(x_cat)
        h, h_perm = torch.split(h, split_size_or_sections=int(h.size()[0] / 2),
                                dim=0)

        z_hat = h[:, 0, :]

        return z_hat
