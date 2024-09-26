import torch.nn as nn
from torch.nn import functional as F

from crc.ood_estimation.base_estimator import OODEstimator


class MLPOODEstimator(OODEstimator):
    def __init__(self, seed, task, data_root):
        super().__init__(seed, task, data_root)

        # Build model
        self.model = self._build_model()

    def _build_model(self):
        """
        Conv NN adapted from P. Lippe.
        """
        return MLPModule(in_channels=3, n_conv_layers=2, h_dim=64)

    def train(self, X, y):
        # Build dataset and dataloader from training data
        pass

    def predict(self, X_ood):
        pass


class MLPModule(nn.Module):
    def __init__(self, in_channels, n_conv_layers, h_dim, z_dim=5):
        super().__init__()

        self.in_channels = in_channels
        self.n_conv_layers = n_conv_layers
        self.h_dim = h_dim
        self.z_dim = z_dim

        NormLayer = lambda d: nn.GroupNorm(num_groups=8, num_channels=d)

        conv_layers = [
            nn.Sequential(
                nn.Conv2d(self.in_channels if i_layer == 0 else self.h_dim,
                          self.h_dim,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False),
                NormLayer(self.h_dim),
                nn.SiLU(),
                nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, stride=1, padding=1,
                          bias=False),
                NormLayer(self.h_dim),
                nn.SiLU
            ) for i_layer in range(self.n_conv_layers)
        ]

        self.network = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(4 * 4 * self.h_dim, 4 * self.h_dim),
            nn.LayerNorm(4 * self.h_dim),
            nn.SiLU(),
            nn.Linear(4 * self.h_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
            nn.SiLU()
        )

        self.lin_head = nn.Linear(self.z_dim, 1)

    def forward(self, x):
        z_hat = self.network(x)
        y_hat = self.lin_head(z_hat)

        return y_hat

