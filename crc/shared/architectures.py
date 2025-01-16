import torch.nn as nn


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


class ResidualBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return nn.Identity(x) + self.net(x)
