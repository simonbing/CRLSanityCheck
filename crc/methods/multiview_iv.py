import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Subset, DataLoader
import wandb

from crc.methods import CRLMethod
from crc.methods.shared.torch_datasets import ChambersDatasetMultiviewOLD, \
    ChambersDatasetMultiview, ChambersDatasetMultiviewSemisynthetic, ChambersDatasetMultiviewSynthetic
from crc.methods.shared import FCEncoder, ConvEncoder
from crc.methods.shared.utils import construct_invertible_mlp
from crc.methods.shared.losses import infonce_loss
from crc.methods.shared.utils import gumbel_softmax_mask
from crc.utils.chamber_sim.simulators.lt.image import DecoderSimple


class Multiview(CRLMethod):
    def __init__(self, in_dims, encoder, selection, tau, **kwargs):
        super().__init__(**kwargs)

        self.encoders = []
        for in_dim, view_encoder in zip(in_dims, encoder):
            match view_encoder:
                case 'fc':
                    self.encoders.append(FCEncoder(in_dim=in_dim,
                                                   latent_dim=self.d,
                                                   hidden_dims=[64, 256, 256, 256, 256, 64]))
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
        match self.dataset_name:
            case 'lt_camera_v1':
                dataset = ChambersDatasetMultiview(dataset=self.dataset_name,
                                                   task=self.task,
                                                   data_root=self.data_root,
                                                   include_iv_data=True)
            case 'multiview_semi_synthetic_mlp':
                mlp_list = [
                    FCEncoder(in_dim=5, latent_dim=20,
                              hidden_dims=[512, 512, 512], relu_slope=0.2,
                              residual=False),
                    FCEncoder(in_dim=3, latent_dim=3,
                              hidden_dims=[512, 512], relu_slope=0.2,
                              residual=False),
                    FCEncoder(in_dim=1, latent_dim=1,
                              hidden_dims=[512, 512], relu_slope=0.2,
                              residual=False),
                    FCEncoder(in_dim=1, latent_dim=1,
                              hidden_dims=[512, 512], relu_slope=0.2,
                              residual=False)
                ]
                dataset = ChambersDatasetMultiviewSemisynthetic(
                    dataset='lt_camera_v1',
                    task=self.task,
                    data_root=self.data_root,
                    include_iv_data=True,
                    transform_list=mlp_list
                )
            case 'multiview_semi_synthetic_decoder':
                decoder_sim = DecoderSimple()
                tf_list = [
                    decoder_sim.simulate_from_inputs,
                    construct_invertible_mlp(n=3,
                                             n_layers=3,
                                             n_iter_cond_thresh=25000,
                                             cond_thresh_ratio=0.001),
                    construct_invertible_mlp(n=1,
                                             n_layers=3,
                                             n_iter_cond_thresh=25000,
                                             cond_thresh_ratio=0.001),
                    construct_invertible_mlp(n=1,
                                             n_layers=3,
                                             n_iter_cond_thresh=25000,
                                             cond_thresh_ratio=0.001)
                ]
                dataset = ChambersDatasetMultiviewSemisynthetic(
                    dataset='lt_camera_v1',
                    task=self.task,
                    data_root=self.data_root,
                    include_iv_data=True,
                    transform_list=tf_list
                )
            case 'multiview_synthetic':
                dataset = ChambersDatasetMultiviewSynthetic(d=5, n=100000)
            case 'multiview_synthetic_2':
                dataset = ChambersDatasetMultiviewSynthetic(d=3, n=100000, gt_model='model_2')
            case 'multiview_synthetic_reprod':
                dataset = ChambersDatasetMultiviewSynthetic(d=6, n=100000, gt_model='reprod')
            case 'multiview_synthetic_chambers_indep':
                dataset = ChambersDatasetMultiviewSynthetic(d=5, n=100000, gt_model='chamber_synth_indep')
            case 'multiview_synthetic_chambers_scm':
                dataset = ChambersDatasetMultiviewSynthetic(d=5, n=100000, gt_model='chamber_synth_scm')
            case _:
                dataset = ChambersDatasetMultiviewOLD(dataset=self.dataset_name,
                                                      task=self.task,
                                                      data_root=self.data_root,
                                                      n_envs=self.n_envs)

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

    def encode_step(self, data):
        data = [x.to(self.device) for x in data[:-1]]

        z = self.model(data)
        return z


class MultiviewModule(nn.Module):
    def __init__(self, encoders):
        super().__init__()

        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        z_list = []

        for x_view, encoder in zip(x, self.encoders):
            z_list += [encoder(x_view)]

        z = torch.stack(z_list)

        return z
