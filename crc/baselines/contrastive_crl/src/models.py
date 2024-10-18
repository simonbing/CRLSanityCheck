
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from crc.baselines.contrastive_crl.src.nonlinearities import Linear_Nonlinearity


def build_model_from_kwargs(model_kwargs):
    if model_kwargs["type"] == 'contrastive':
        if model_kwargs.get("image", False):
            return get_contrastive_image(**model_kwargs)
        else:
            return get_contrastive_synthetic(**model_kwargs)
    elif model_kwargs["type"] == 'vae':
        if model_kwargs.get("image", False):
            return get_vae_image(**model_kwargs)
        else:
            return get_vae_synthetic(**model_kwargs)
    elif model_kwargs["type"] in ['vae_vanilla', 'vae_vanilla2']:
        if model_kwargs.get("image", False):
            return get_vae_image(vanilla=True, **model_kwargs)
        else:
            return get_vae_synthetic(vanilla=True, **model_kwargs)
    elif model_kwargs["type"] == 'vae_contrastive':
        return get_vae_contrastive(**model_kwargs)
    elif model_kwargs["type"] == 'oracle':
        return get_oracle_synthetic(**model_kwargs)
    elif model_kwargs["type"] == 'contrastive_linear':
        return get_contrastive_linear_synthetic(**model_kwargs)
    else:
        raise NotImplementedError('Must be in contrastive or vae')

def get_contrastive_synthetic(input_dim, latent_dim, hidden_dim, hidden_layers=0, residual=True, **kwargs):
    embedding = EmbeddingNet(input_dim, latent_dim, hidden_dim, hidden_layers=hidden_layers, residual=residual)
    return ContrastiveModel(latent_dim, embedding)


def get_contrastive_linear_synthetic(input_dim, latent_dim, hidden_dim, hidden_layers=0, residual=True, **kwargs):
    embedding = Linear_Nonlinearity(input_dim=input_dim, output_dim=latent_dim)
    return ContrastiveModel(latent_dim, embedding)


def get_contrastive_image(latent_dim, conv, channels=1, **kwargs):
    if conv:
        embedding = ImageEncoderChambers(latent_dim)
    else:
        embedding = ImageEncoderMinimal(latent_dim, vae=False, channels=channels)
    return ContrastiveModel(latent_dim, embedding)


def get_oracle_synthetic(input_dim, latent_dim, **kwargs):
    embedding = OracleEmbedding(input_dim, latent_dim)
    return ContrastiveModel(latent_dim, embedding)


def get_vae_synthetic(input_dim, latent_dim, hidden_dim, hidden_layers=0, vanilla=False, **kwargs):
    encoder = Encoder(input_dim, latent_dim, hidden_dim, hidden_layers=hidden_layers)
    decoder = Decoder(latent_dim, input_dim, hidden_dim, hidden_layers=hidden_layers)
    return VAE_DAG(latent_dim, encoder, decoder, vanilla=vanilla)


def get_vae_image(input_dim, latent_dim, hidden_dim, hidden_layers=0, vanilla=False, **kwargs):
    if vanilla:
        encoder = EncoderVAE(latent_dim)
        decoder = DecoderVAE(latent_dim)
    else:
        encoder = ImageEncoderMinimal(latent_dim, vae=True)
        decoder = ImageDecoderMinimal(latent_dim)
    return VAE_DAG(latent_dim, encoder, decoder, vanilla=vanilla)


def get_vae_contrastive(input_dim, latent_dim, hidden_dim, hidden_layers=0, residual=True, vae_latent_dim=25, **kwargs):
    encoder = EncoderVAE(vae_latent_dim)
    decoder = DecoderVAE(vae_latent_dim)
    embedding = EmbeddingNet(vae_latent_dim, latent_dim, hidden_dim, hidden_layers=hidden_layers, residual=residual)
    return VAE_Contrastive(latent_dim, encoder, decoder, embedding)


class ContrastiveModel(nn.Module):
    def __init__(self, d, embedding):
        super(ContrastiveModel, self).__init__()
        self.embedding = embedding
        self.d = d
        self.parametric_part = ParametricPart(d)

    def get_z(self, x):
        return self.embedding(x)

    def forward(self, x, t, return_embedding=False):
        z = self.embedding(x)
        logit = self.parametric_part(z, t)
        if return_embedding:
            return logit, z
        return logit


class VAE_Contrastive(nn.Module):
    def __init__(self, d, encoder, decoder, embedding):
        super(VAE_Contrastive, self).__init__()
        self.d = d
        self.parametric_part = ParametricPart(d)
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.match_observation_dist_only = False

    def forward(self, x, t, reparametrize=True):
        z_est, log_var = self.encoder(x)
        z_est_detach = z_est.detach().clone()
        z = self.embedding(z_est_detach)
        logits = self.parametric_part(z, t)
        if reparametrize:
            eps = self.reparameterization(z_est, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        else:
            eps = z_est
        x_hat = self.decoder(eps)
        return x_hat, z_est, log_var, logits

    @staticmethod
    def reparameterization(mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def get_eps_z(self, x):
        z_est, _ = self.encoder(x)
        return z_est, z_est

    def get_z(self, x):
        z_est, _ = self.encoder(x)
        z_est_detach = z_est.detach()
        return self.embedding(z_est_detach)

    def get_latents(self, x):
        z_est, _ = self.encoder(x)
        return z_est


class VAE_DAG(nn.Module):
    def __init__(self, d, encoder, decoder, vanilla=False):
        super(VAE_DAG, self).__init__()
        self.d = d
        self.vanilla = vanilla
        if vanilla:
            self.parametric_part = TrivialParametricPart(d)
        else:
            self.parametric_part = ParametricPart(d)
        self.encoder = encoder
        self.decoder = decoder
        self.match_observation_dist_only = True

    def forward(self, x, t, reparametrize=True):
        z_est, log_var = self.encoder(x)
        eps_est = torch.matmul(z_est, self.parametric_part.get_B())
        logits = self.parametric_part(z_est, t)
        if reparametrize:
            eps = self.reparameterization(eps_est, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        else:
            eps = eps_est
        x_hat = self.decoder(eps)
        return x_hat, eps_est, log_var, logits

    @staticmethod
    def reparameterization(mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def get_eps_z(self, x):
        z_est, _ = self.encoder(x)
        eps = torch.matmul(z_est, self.parametric_part.get_B())
        return eps, z_est

    def get_z(self, x):
        z_est, _ = self.encoder(x)
        return z_est

    def get_latents(self, x):
        z_est, _ = self.encoder(x)
        return z_est


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, hidden_layers=0):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.FC_input(x))
        for lin_layer in self.hidden_layers:
            x = self.LeakyReLU(lin_layer(x))
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, hidden_layers=0):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.FC_hidden(x))
        for lin_layer in self.hidden_layers:
            x = self.LeakyReLU(lin_layer(x))
        x_hat = self.FC_output(x)
        return x_hat


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, hidden_layers=0, residual=True):
        super(EmbeddingNet, self).__init__()
        self.residual = residual
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.FC_out = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.FC_input(x))
        for lin_layer in self.hidden_layers:
            if self.residual:
                x = self.LeakyReLU(lin_layer(x)) + x
            else:
                x = self.LeakyReLU(lin_layer(x))
        return self.FC_out(x)


class ParametricPart(nn.Module):
    def __init__(self, d):
        super(ParametricPart, self).__init__()
        self.d = d
        # The intercept for the logistic regression t_i vs obs
        self.intercepts = torch.nn.Parameter(torch.ones(d, requires_grad=True))
        # The linear terms (they are by assumption aligned with e_i)
        self.shifts = torch.nn.Parameter(torch.zeros(d, requires_grad=True))
        # the scaling matrix D
        self.lambdas = torch.nn.Parameter(torch.ones(d, requires_grad=True))
        # self.scales = torch.nn.Parameter(torch.ones((1, d), requires_grad=True))
        self.scales = torch.nn.Parameter(torch.ones((1, d), requires_grad=True))
        self.A = torch.nn.Parameter(torch.eye(d, requires_grad=True))
        # to avoid training the diagonal of A we remove it
        self.register_buffer('exclusion_matrix', torch.ones((d, d)) - torch.eye(d))

    def get_B(self):
        return self.scales.view(-1, 1) * (torch.eye(self.d, device=self.A.device) - self.A * self.exclusion_matrix)

    def forward(self, z, t):
        z_sel = z[torch.arange(z.size(0)), t]
        intercepts_sel = self.intercepts[t]
        lambdas_sel = self.lambdas[t]
        shifts_sel = self.shifts[t]
        B = self.get_B()
        trafo_sel = B[t]
        # TODO check sign carefully!!! (Should be fine now)
        logit = shifts_sel + z_sel * intercepts_sel + (z_sel * lambdas_sel) ** 2 - torch.sum(z * trafo_sel, dim=1) ** 2
        logit = torch.cat((logit.view(-1, 1), torch.zeros((z.size(0), 1), device=logit.device)), dim=1)
        return logit


class TrivialParametricPart(nn.Module):
    def __init__(self, d):
        super(TrivialParametricPart, self).__init__()
        self.d = d
        self.register_buffer('A', torch.zeros((d, d)))

    def get_B(self):
        return torch.eye(self.d, device=self.A.device)

    def forward(self, z, t):
        return torch.zeros((z.size(0), 2), device=z.device)

class OracleEmbedding(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(OracleEmbedding, self).__init__()
        self.register_buffer('fc', torch.zeros(input_dim, latent_dim))

    def forward(self, x):
        return torch.matmul(x, self.fc)


class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.base_architecture = 'resnet18'
        self.width = 128

        self.base_model = resnet18(pretrained=True)
        self.feat_layers = list(self.base_model.children())[:-1]
        self.feat_net = nn.Sequential(*self.feat_layers)

        self.fc_layers = [
            nn.Linear(512, self.width),
            nn.LeakyReLU(),
            nn.Linear(self.width, self.latent_dim),
        ]

        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.feat_net(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc_net(x)
        return x


class ImageEncoderChambers(nn.Module):
    def __init__(self, latent_dim, n_conv_layers=2):
        super().__init__()

        NormLayer = lambda d: nn.GroupNorm(num_groups=8, num_channels=d)

        h_dim = 64


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

    def forward(self, x):
        x = self.conv_encoder(x)

        return x


class ImageEncoderMinimal(torch.nn.Module):
    def __init__(self, latent_dim, vae=False, channels=1):
        super(ImageEncoderMinimal, self).__init__()

        self.latent_dim = latent_dim
        self.vae = vae
        self.conv1 = nn.Conv2d(3, channels, 5, stride=3)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(4, 4, 5)

        self.fc_layers = [
            nn.Linear(100*channels, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.latent_dim * 2),
        ]
        # self.fc_layers = [nn.Linear(100, self.latent_dim)]

        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape((x.size(0), -1))
        x = self.fc_net(x)
        if self.vae:
            return x[:, :self.latent_dim], x[:, self.latent_dim:]
        return x[:, :self.latent_dim]


class ImageDecoderMinimal(torch.nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoderMinimal, self).__init__()

        self.latent_dim = latent_dim
        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 5, 4, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(32, 3, 4, 3, 0, bias=False),
            nn.Sigmoid()
        )

        self.fc_layers = [
            nn.Linear(self.latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3 * 6 ** 2),
        ]
        # self.fc_layers = [nn.Linear(100, self.latent_dim)]

        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        x = self.fc_net(x)
        x = x.reshape((x.size(0), 3, 6, 6))
        return self.deconvolutions(x)


class EncoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        # Latent space layers
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    def forward(self, x):
        return self.encode(x)

class DecoderVAE(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderVAE, self).__init__()

        self.latent_dim = latent_dim
        self.channels = 32
        self.register_buffer('scales', torch.zeros(latent_dim))
        # self.scales = nn.Parameter(torch.zeros(latent_dim))
        self.fc = nn.Linear(latent_dim, 9 * latent_dim * self.channels * 4)
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 4 * self.channels, 4 * self.channels, kernel_size=4, stride=2, padding=0),
            # nn.BatchNorm2d(self.channels * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4 * self.channels, 2 * self.channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.channels * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2 * self.channels, self.channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure outputs are between 0 and 1
        )

    def decode(self, z):
        z = self.fc(z)
        z = z.reshape(z.size(0), self.latent_dim * self.channels * 4, 3, 3)  # Reshape tensor
        x = self.decoder(z)
        return x

    def forward(self, x):
        return self.decode(x)
