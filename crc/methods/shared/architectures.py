"""Shared architectural compenents."""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FCEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dims, residual=False,
                 relu_slope=0.01):
        super().__init__()

        if not residual:
            encoder_layers = [
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]),
                    nn.LeakyReLU(relu_slope)
                ) for i in range(len(hidden_dims))
            ]
        else:
            encoder_layers = [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[0]),
                    nn.LeakyReLU(relu_slope),
                    *[ResidualBlock(
                        nn.Sequential(
                            nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                            nn.LeakyReLU(relu_slope)
                        )
                    ) for i in range(1, len(hidden_dims))]
                )
            ]

        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.Linear(hidden_dims[-1], latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=4, stride=2, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=4, stride=2, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=4, stride=2, out_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, kernel_size=4, stride=2, out_channels=64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim),
        )

        # ResNET Encoder
        # self.encoder = nn.Sequential(
        #     resnet18(num_classes=100),
        #     nn.LeakyReLU(),
        #     nn.Linear(100, latent_dim)
        # )

    def forward(self, x):
        return self.encoder(x)


class ResidualBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return nn.Identity(x) + self.net(x)
