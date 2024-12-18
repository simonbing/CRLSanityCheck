import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset, DataLoader
import wandb

from crc.methods import CRLMethod
from crc.methods.shared.torch_datasets import ChambersDatasetMultiviewOLD, ChambersDatasetMultiview
from crc.methods.shared import FCEncoder, ConvEncoder
from crc.methods.shared.losses import infonce_loss
from crc.methods.shared.utils import gumbel_softmax_mask


class Multiview(CRLMethod):
    def __init__(self, in_dims, encoder, selection, tau, **kwargs):
        super().__init__(**kwargs)

        self.encoders = []
        for in_dim, view_encoder in zip(in_dims, encoder):
            match view_encoder:
                case 'fc':
                    self.encoders.append(FCEncoder(in_dim=in_dim,
                                                   latent_dim=self.d,
                                                   hidden_dims=[512, 512, 512]))
                case 'conv':
                    self.encoders.append(ConvEncoder(latent_dim=self.d))
                case _:
                    ValueError, 'Invalid encoder type passed!'

        self._build_model()
        self.optimizer = self._get_optimizer()

        # self.n_envs = n_envs
        self.selection = selection
        self.tau = tau

    def _build_model(self):
        self.model = MultiviewModule(encoders=self.encoders).to(self.device)

    def get_dataset(self):
        match self.dataset:
            case 'lt_camera_v1':
                dataset = ChambersDatasetMultiview(dataset=self.dataset,
                                                   task=self.task,
                                                   data_root=self.data_root,
                                                   include_iv_data=True)
            case _:
                dataset = ChambersDatasetMultiviewOLD(dataset=self.dataset,
                                                      task=self.task,
                                                      data_root=self.data_root,
                                                      n_envs=self.n_envs)
        self.subsets = dataset.subsets
        self.content_indices = dataset.content_indices

        return dataset

    def loss_f(self, z, estimated_content_indices, subsets):
        return infonce_loss(hz=z,
                            sim_metric=torch.nn.CosineSimilarity(dim=-1),
                            criterion=torch.nn.CrossEntropyLoss(),
                            projector=(lambda x: x),
                            tau=self.tau,
                            estimated_content_indices=estimated_content_indices,
                            subsets=subsets)

    def train_step(self, data):
        data = [x.to(self.device) for x in data]

        # Get encoding
        z = self.model(data)  # (n_views, batch_size, d)

        # Estimate content indices
        if self.selection == 'ground_truth':
            estimated_content_indices = self.content_indices
        else:
            avg_logits = z.reshape(-1, z.shape[-1]).mean(0)[None]
            content_sizes = [len(content) for content in
                             self.content_indices]

            content_masks = gumbel_softmax_mask(avg_logits=avg_logits,
                                                subsets=self.subsets,
                                                content_sizes=content_sizes)
            estimated_content_indices = []
            for c_mask in content_masks:
                c_ind = torch.where(c_mask)[-1].tolist()
                estimated_content_indices += [c_ind]

        loss = self.loss_f(z, estimated_content_indices,
                           self.subsets)

        return loss, {'loss': loss.item()}

    def encode_step(self, data):
        pass


class MultiviewModule(nn.Module):
    def __init__(self, encoders):
        super().__init__()
        # Don't have a different encoder for each view, assume they are shared
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        z_list = []

        for x_view, encoder in zip(x, self.encoders):
            z_list += [encoder(x_view)]

        z = torch.stack(z_list)

        return z
