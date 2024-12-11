import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset, DataLoader
import wandb

from crc.methods import CRLMethod
from crc.methods.shared import ChambersDatasetMultiview
from crc.methods.shared import FCEncoder, ConvEncoder
from crc.methods.shared.losses import infonce_loss
from crc.methods.shared.utils import gumbel_softmax_mask
from crc.utils import train_val_test_split


class MultiviewIv(CRLMethod):
    def __init__(self, n_envs, encoder, selection, tau, **kwargs):
        super().__init__(**kwargs)

        match encoder:
            case 'fc':
                self.encoder = FCEncoder()
            case 'conv':
                self.encoder = ConvEncoder(latent_dim=self.d)
            case _:
                ValueError, 'Invalid encoder type passed!'

        self._build_model()
        self.optimizer = self._get_optimizer()

        self.n_envs = n_envs
        self.selection = selection
        self.tau = tau

        self.dataset = self._get_dataset()

    def _build_model(self):
        self.model = MultiviewIvModule(encoder=self.encoder).to(self.device)

    def _get_dataset(self):
        match self.dataset:
            case _:
                return ChambersDatasetMultiview(dataset=self.dataset,
                                                task=self.task,
                                                data_root=self.data_root,
                                                n_envs=self.n_envs)

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
            estimated_content_indices = self.dataset.content_indices
        else:
            avg_logits = z.reshape(-1, z.shape[-1]).mean(0)[None]
            content_sizes = [len(content) for content in
                             self.dataset.content_indices]

            content_masks = gumbel_softmax_mask(avg_logits=avg_logits,
                                                subsets=self.dataset.subsets,
                                                content_sizes=content_sizes)
            estimated_content_indices = []
            for c_mask in content_masks:
                c_ind = torch.where(c_mask)[-1].tolist()
                estimated_content_indices += [c_ind]

        loss = self.loss_f(z, estimated_content_indices,
                           self.dataset.subsets)

        return loss, {'loss': loss.item()}

    # def train(self):
    #     # Get train, val, test dataset
    #     dataset = self._get_dataset()
    #
    #     train_idxs, val_idxs, test_idxs = train_val_test_split(np.arange(len(dataset)),
    #                                                            train_size=self.train_size,
    #                                                            val_size=self.val_size,
    #                                                            random_state=self.seed)
    #
    #     train_dataset = Subset(dataset, train_idxs)
    #     val_dataset = Subset(dataset, val_idxs)
    #
    #     train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
    #     val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size)
    #
    #     best_val_loss = np.inf
    #     best_model = copy.deepcopy(self.model)
    #
    #     # Train loop
    #     for i in range(self.epochs):
    #         self.model.train()
    #         # Minibatch training
    #         train_loss_values = []
    #         for data in train_dataloader:
    #             # Zero gradients
    #             self.optimizer.zero_grad()
    #
    #             data = [x.to(self.device) for x in data]
    #
    #             # Get encoding
    #             z = self.model(data)  # (n_views, batch_size, d)
    #
    #             # Estimate content indices
    #             if self.selection == 'ground_truth':
    #                 estimated_content_indices = train_dataset.dataset.content_indices
    #             else:
    #                 avg_logits = z.reshape(-1, z.shape[-1]).mean(0)[None]
    #                 content_sizes = [len(content) for content in train_dataset.dataset.content_indices]
    #
    #                 content_masks = gumbel_softmax_mask(avg_logits=avg_logits,
    #                                                     subsets=train_dataset.dataset.subsets,
    #                                                     content_sizes=content_sizes)
    #                 estimated_content_indices = []
    #                 for c_mask in content_masks:
    #                     c_ind = torch.where(c_mask)[-1].tolist()
    #                     estimated_content_indices += [c_ind]
    #
    #             loss = self.loss_f(z, estimated_content_indices, train_dataset.dataset.subsets)
    #             train_loss_values.append(loss.item())
    #
    #             wandb.log({'train_loss': loss.item()})
    #
    #             # Apply gradients
    #             loss.backward()
    #             clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
    #             self.optimizer.step()
    #
    #         # Validation
    #         if (i+1) % self.val_step == 0 or i == (self.epochs-1):
    #             self.model.eval()
    #             val_loss_values = []
    #             for data in val_dataloader:
    #                 data = [x.to(self.device) for x in data]
    #
    #                 # Get encoding
    #                 z = self.model(data)  # (n_views, batch_size, d)
    #                 # Estimate content indices
    #                 if self.selection == 'ground_truth':
    #                     estimated_content_indices = train_dataset.dataset.content_indices
    #                 else:
    #                     avg_logits = z.reshape(-1, z.shape[-1]).mean(0)[None]
    #                     content_sizes = [len(content) for content in
    #                                      train_dataset.dataset.content_indices]
    #
    #                     content_masks = gumbel_softmax_mask(
    #                         avg_logits=avg_logits,
    #                         subsets=train_dataset.subsets,
    #                         content_sizes=content_sizes)
    #                     estimated_content_indices = []
    #                     for c_mask in content_masks:
    #                         c_ind = torch.where(c_mask)[-1].tolist()
    #                         estimated_content_indices += [c_ind]
    #
    #                 loss = self.loss_f(z, estimated_content_indices,
    #                                    train_dataset.dataset.subsets)
    #                 wandb.log({'val_loss': loss.item()})
    #
    #                 val_loss_values.append(loss.item())
    #
    #                 if np.mean(val_loss_values) <= best_val_loss:
    #                     best_val_loss = np.mean(val_loss_values)
    #                     best_model = copy.deepcopy(self.model)
    #
    #                 # log loss
    #                 # log losses
    #
    #     return best_model


class MultiviewIvModule(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # Don't have a different encoder for each view, assume they are shared
        self.joint_encoder = encoder

    def forward(self, x):
        z_list = []

        for x_view in x:
            z_list += [self.joint_encoder(x_view)]

        z = torch.stack(z_list)

        return z
