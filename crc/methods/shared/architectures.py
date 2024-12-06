"""Shared architectural compenents."""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FCEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Flatten input
        pass


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
