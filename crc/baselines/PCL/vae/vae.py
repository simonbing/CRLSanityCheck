"""VAE model"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
# import torchvision
import numpy as np

from subfunc.showdata import *

torch.manual_seed(0)


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
class VariationalAutoencoder(nn.Module):
    def __init__(self, h_sizes, num_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(h_sizes, num_dim)
        self.decoder = Decoder(h_sizes, num_dim)

    def forward(self, x):
        z, mu = self.encoder(x)
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    def __init__(self, h_sizes, num_dim, pool_size=2):
        super(VariationalEncoder, self).__init__()

        h_sizes_aug = [num_dim] + h_sizes
        h = [nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k]*pool_size) for k in range(1, len(h_sizes_aug)-1)]
        self.h = nn.ModuleList(h)
        self.h_mu = nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1])
        self.h_sigma = nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1])
        self.maxout = Maxout(pool_size)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        h = x
        for k in range(len(self.h)):
            h = self.h[k](h)
            h = self.maxout(h)
        mu = self.h_mu(h)
        sigma = torch.exp(self.h_sigma(h))
        # z = mu
        z = mu + sigma*self.N.sample(mu.shape).to(x.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, mu


class Decoder(nn.Module):
    def __init__(self, h_sizes, num_dim, pool_size=2):
        super(Decoder, self).__init__()
        h_sizes_aug = [num_dim] + h_sizes
        h = [nn.Linear(h_sizes_aug[k-1], h_sizes_aug[k]*pool_size) for k in range(1, len(h_sizes_aug)-1)]
        h.append(nn.Linear(h_sizes_aug[-2], h_sizes_aug[-1], bias=False))
        self.h = nn.ModuleList(h)
        self.maxout = Maxout(pool_size)

    def forward(self, z):
        for k in range(len(self.h)):
            z = self.h[k](z)
            if k != len(self.h) - 1:
                z = self.maxout(z)
        return z

