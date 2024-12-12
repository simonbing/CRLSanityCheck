import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from crc.methods import CRLMethod
from crc.methods.shared import ChambersDatasetContrastive
from crc.methods.shared import FCEncoder, ConvEncoder


class ContrastCRL(CRLMethod):
    def __init__(self, encoder, kappa, eta, mu, **kwargs):
        super().__init__(**kwargs)

        match encoder:
            case 'fc':
                self.encoder = FCEncoder()
            case 'conv':
                self.encoder = ConvEncoder(latent_dim=self.d)
            case _:
                ValueError, 'Invalid encoder type passed!'

        self.kappa = kappa
        self.eta = eta
        self.mu = mu

        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.notears_loss = lambda A: torch.trace(torch.matrix_exp(A * A) - A.size(0))

        self.model = self._build_model()

        self.optimizer = self._get_optimizer()

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.3, patience=3)

    def _build_model(self):
        return ContrastCRLModule(d=self.d, encoder=self.encoder)

    def get_dataset(self):
        match self.dataset:
            case _:
                return ChambersDatasetContrastive(dataset=self.dataset,
                                                  task=self.task,
                                                  data_root=self.data_root)

    def train_step(self, data):
        X_obs, X_iv, iv_idx = data
        X_obs = X_obs.to(self.device)
        X_iv = X_iv.to(self.device)
        iv_idx = iv_idx.to(self.device)

        logits_obs, z_obs = self.model(X_obs, iv_idx, return_z=True)
        logits_iv = self.model(X_iv, iv_idx)

        method_specific_loss = self.kappa * torch.sum(torch.mean(z_obs, dim=0) ** 2)

        classifier_loss = self.ce_loss(logits_obs,torch.zeros(X_obs.size(0),
                                                              dtype=torch.long,
                                                              device=self.device)) \
                          + self.ce_loss(logits_iv,torch.ones(X_iv.size(0),
                                                              dtype=torch.long,
                                                              device=self.device))

        reg_loss = self.eta * torch.sum(torch.abs(self.model.A)) \
                   + self.mu * self.notears_loss(self.model.A)

        loss = method_specific_loss + classifier_loss + reg_loss

        return loss, {'total_loss': loss.item(),
                      'method_loss': method_specific_loss.item(),
                      'classifier_loss': classifier_loss.item(),
                      'reg_loss': reg_loss.item()}

    def encode_step(self, data):
        X_obs = data[0]
        X_obs = X_obs.to(self.device)

        z = self.model.get_z(X_obs)

        return z


class ContrastCRLModule(nn.Module):
    def __init__(self, d, encoder):
        super().__init__()

        self.d = d
        self.encoder = encoder
        # Final parametric (classification) layer
        # The intercept for the logistic regression t_i vs obs
        self.intercepts = torch.nn.Parameter(torch.ones(d, requires_grad=True))
        # The linear terms (they are by assumption aligned with e_i)
        self.shifts = torch.nn.Parameter(torch.zeros(d, requires_grad=True))
        # the scaling matrix D
        self.lambdas = torch.nn.Parameter(torch.ones(d, requires_grad=True))
        self.scales = torch.nn.Parameter(torch.ones((1, d), requires_grad=True))

        self.A = torch.nn.Parameter(torch.eye(d, requires_grad=True))
        # to avoid training the diagonal of A we remove it
        self.register_buffer('exclusion_matrix', torch.ones((d, d)) - torch.eye(d))

    def get_B(self):
        return self.scales.view(-1, 1) * (torch.eye(self.d, device=self.A.device) - self.A * self.exclusion_matrix)

    def get_z(self, x):
        return self.encoder(x)

    def forward(self, x, k, return_z=False):
        z = self.get_z(x)

        z_sel = z[torch.arange(z.size(0)), k]
        intercepts_sel = self.intercepts[k]
        lambdas_sel = self.lambdas[k]
        shifts_sel = self.shifts[k]
        B = self.get_B()
        B_sel = B[k]

        logits = shifts_sel + z_sel * intercepts_sel + (z_sel * lambdas_sel) ** 2 - torch.sum(z * B_sel, dim=1) ** 2
        logits = torch.cat((logits.view(-1, 1), torch.zeros((z.size(0), 1), device=logits.device)), dim=1)

        if return_z:
            return logits, z
        else:
            return logits


