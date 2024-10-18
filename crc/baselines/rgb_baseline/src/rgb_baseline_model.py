import torch


class RGBBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=(2,3))

    def get_z(self, x):
        z_hat = self.forward(x)

        return z_hat
    