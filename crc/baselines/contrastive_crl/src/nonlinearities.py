import torch
from torchvision import transforms
# import pygame
# import pygame.gfxdraw
import numpy as np

class Linear_Nonlinearity(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_Nonlinearity, self).__init__()
        self.A = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        return self.A(x)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.ident = torch.nn.Identity()

    def forward(self, x):
        return self.ident(x)


COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
    [255, 140, 0],
    [34, 139, 34],
    [255, 69, 0],
    [210, 105, 30],
    [255, 99, 71],
    [128, 0, 128],
    [0, 128, 128],
    [154, 205, 50]
]

SCREEN_DIM = 64

class ImageGenerator:
    """
    The code to generate the images is extracted from https://github.com/facebookresearch/CausalRepID which is the
    repository for the paper https://arxiv.org/abs/2209.11924.
    """
    ball_rad = 2.0 * 0.04
    screen_dim = 64

    def __init__(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.data_transform=  transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        x = x.detach().numpy()
        return self.generate_images(x)

    def generate_images(self, x):
        images = []
        one_dimensional = x.ndim == 1
        if one_dimensional:
            x = np.expand_dims(x, 0)
        assert x.shape[1] % 2 == 0, "Can only create scenes with even number of latents (2 per ball)"
        n_balls = x.shape[1] // 2
        for i in range(x.shape[0]):
            z = x[i]
            z = z.reshape(n_balls, 2)
            z = z + .5 # offset to center in the image
            images.append(self.draw_scene(z))

        images = torch.tensor(np.stack(images)).float()
        images = images.permute(0, 3, 1, 2)
        images = 1 - images / 255.0
        if one_dimensional:
            images = np.squeeze(images, 0)
        return images

    def draw_scene(self, z):
        self.surf.fill((255, 255, 255))
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(z.shape[0]):
            circle(
                z[i, 0],
                z[i, 1],
                self.surf,
                color=COLOURS_[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    pygame.gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    pygame.gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
