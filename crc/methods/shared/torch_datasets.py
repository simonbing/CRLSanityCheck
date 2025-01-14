import inspect
import os

from causalchamber.datasets import Dataset as ChamberData
import numpy as np
import pandas as pd
from sempler.generators import dag_avg_deg
from sempler import LGANM
from skimage import io
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from crc.utils import get_task_environments
from crc.methods.shared.architectures import FCEncoder
from crc.methods.shared.utils import construct_invertible_mlp
from crc.baselines.contrastive_crl.src.models import EmbeddingNet


def _map_iv_envs(idx, exp, env_list):
    map = [f'{exp}_{env}' for env in env_list]

    return map[idx]


class ChambersDatasetContrastive(Dataset):
    def __init__(self, dataset, task, data_root):
        super().__init__()
        self.eval = False

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)

        chamber_data = ChamberData(name=self.chamber_data_name,
                                   root=self.data_root,
                                   download=True)

        # Observational data
        obs_data = chamber_data.get_experiment(
            name=f'{self.exp}_reference').as_pandas_dataframe()
        Z_obs = obs_data[self.features].to_numpy()
        # Normalize
        self.means = np.mean(Z_obs, axis=0, keepdims=True)
        self.scale_factors = np.std(Z_obs, axis=0, keepdims=True)
        Z_obs = (Z_obs - self.means) / self.scale_factors
        self.Z_obs = np.tile(Z_obs, (Z_obs.shape[-1], 1))

        # Interventional data
        iv_data_list = [chamber_data.get_experiment(
            name=f'{self.exp}_{env}').as_pandas_dataframe() for env in
                        self.env_list]

        # Enforce that all iv_data have the same length
        n_list = [len(df) for df in iv_data_list]
        n_min = min(n_list)
        iv_data_list = [df[:n_min] for df in iv_data_list]

        # Get one big df for all iv data
        self.iv_data = pd.concat(iv_data_list)

        Z_iv = self.iv_data[self.features].to_numpy()
        self.Z_iv = (Z_iv - self.means) / self.scale_factors

        # Generate intervention index list
        iv_names = []
        for idx, iv_data in enumerate(iv_data_list):
            iv_names.append(np.repeat(idx, len(iv_data)))
        self.iv_names = np.concatenate(iv_names)

        # Resample observational data to have same nr of samples as iv_data
        # self.obs_data = obs_data.loc[np.random.choice(len(obs_data),
        #                                               size=len(self.iv_data),
        #                                               replace=True), :]
        self.obs_data = pd.concat([obs_data] * len(self.features))

        # Get ground truth adjacency matrix
        match self.exp:
            case a if a in ('scm_1', 'scm_2'):
                self.W = np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1],
                        [0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
            case 'scm_4':
                self.W = np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ]
                )
            case 'scm_5':
                self.W = np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0],
                    ]
                )

    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, item):
        # Observational sample
        obs_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                    f'{self.exp}_reference',
                                    'images_64',
                                    self.obs_data['image_file'].iloc[item])
        obs_sample = io.imread(obs_img_name)

        # Interventional sample
        iv_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                   _map_iv_envs(self.iv_names[item],
                                                self.exp, self.env_list),
                                   'images_64',
                                   self.iv_data['image_file'].iloc[item])
        iv_sample = io.imread(iv_img_name)

        # Normalize inputs
        obs_sample = obs_sample / 255.0
        iv_sample = iv_sample / 255.0

        # Ground truth variables
        Z_obs = self.obs_data[self.features].iloc[item].to_numpy()
        Z_iv = self.iv_data[self.features].iloc[item].to_numpy()

        if not self.eval:
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32), \
                torch.as_tensor(iv_sample.transpose((2, 0, 1)),
                                dtype=torch.float32), \
                torch.as_tensor(self.iv_names[item],
                                dtype=torch.int)
        else:
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32), \
                Z_obs


class ChambersDatasetContrastiveSynthetic(Dataset):
    def __init__(self, d, k, n=10000, x_dim=20):
        super().__init__()
        self.eval = False

        # Sample adjacency
        W, ordering = dag_avg_deg(p=d, k=k, w_min=0.25, w_max=1.0,
                                  return_ordering=True, random_state=42)
        # Add signs
        rs = np.random.RandomState(42)
        mask = rs.binomial(n=1, p=0.5, size=W.size,).reshape(W.shape)
        # self.W = W - 2 * mask * W
        # self.W = W

        ###
        self.W = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.71943922, 0.0, 0.0, 0.0, 0.67726298],
        [0.0, 0.89303215, 0.0, 0.0, 0.98534901],
        [0.84868401, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]])

        # Observational means
        obs_means = np.array(
            [0.3773956, 0.34388784, 0.38585979, 0.3697368, 0.30941773])
        # Standard deviations
        stds = np.array(
            [0.09926867, 0.09283419, 0.09358193, 0.07384341, 0.08351158])
        int_means = np.array(
            [0.48110359, 0.45315549, 0.49229844, 0.47796442, 0.41385188])
        ###

        variances_obs = rs.uniform(1.0, 2.0, size=d)
        # variances_obs = rs.uniform(0.01, 0.02, size=d)

        lganm = LGANM(W=self.W, means=np.zeros(d), variances=variances_obs, random_state=42)
        # means_obs = rs.uniform(0.1, 0.2, size=d)
        # lganm = LGANM(W=self.W, means=means_obs, variances=variances_obs, random_state=42)
        # lganm = LGANM(W=self.W, means=np.zeros(d), variances=stds,
        #               random_state=42)

        # Observational samples
        obs_samples = lganm.sample(n)
        self.means = np.mean(obs_samples, axis=0, keepdims=True)
        self.stds = np.std(obs_samples, axis=0, keepdims=True)

        obs_samples = (obs_samples - self.means) / self.stds

        # scales_up = np.array([[255/8, 255/8, 255/8, 90/4, 90/4]])
        # shifts_up = np.array([[127.5, 127.5, 127.5, 0, 0]])
        #
        # obs_up = np.clip(obs_samples, a_min=-4, a_max=4) * scales_up + shifts_up

        self.Z_obs = np.tile(obs_samples, (d, 1))

        # Interventional samples
        means_iv = rs.uniform(1.0, 2.0, size=d)
        # means_iv = rs.uniform(0.1, 0.2, size=d)
        # means_iv = int_means
        # mask = rs.binomial(n=1, p=0.5, size=means_iv.size).reshape(means_iv.shape)
        # means_iv = means_iv - 2 * mask * means_iv
        variances_iv = rs.uniform(1.0, 2.0, size=d)

        iv_samples = []
        iv_targets = []
        for i in range(d):
            # samples = lganm.sample(n, do_interventions={i: (means_iv[i], variances_iv[i])})
            samples = lganm.sample(n, do_interventions={i: (means_iv[i], variances_obs[i])})
            # samples = lganm.sample(n, do_interventions={i: (int_means[i], stds[i])})
            iv_samples.append(samples)
            iv_targets.append(i * np.ones(n, dtype=np.int_))
        self.Z_iv = np.concatenate(iv_samples, axis=0)

        self.Z_iv = (self.Z_iv - self.means) / self.stds

        ###
        self.Z_obs = np.clip(self.Z_obs, a_min=-4, a_max=4)
        self.Z_iv = np.clip(self.Z_iv, a_min=-4, a_max=4)
        ###

        self.iv_names = np.concatenate(iv_targets, axis=0)

        # self.transform = FCEncoder(in_dim=d, latent_dim=x_dim,
        #                            hidden_dims=[512, 512, 512],
        #                            relu_slope=0.2,
        #                            residual=False)

        self.transform = EmbeddingNet(5, 20, 512, hidden_layers=3, residual=False)

    def __len__(self):
        return len(self.Z_obs)

    def __getitem__(self, item):
        if not self.eval:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                self.transform(torch.as_tensor(self.Z_iv[item], dtype=torch.float32)), \
                torch.as_tensor(self.iv_names[item], dtype=torch.int)
        else:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                self.Z_obs[item]


class ChambersDatasetContrastiveSemiSynthetic(ChambersDatasetContrastive):
    def __init__(self, dataset, task, data_root, transform):
        super().__init__(dataset, task, data_root)

        self.transform = transform

    def __getitem__(self, item):
        # Z_obs = self.obs_data[self.features].iloc[item].to_numpy()
        # Z_iv = self.iv_data[self.features].iloc[item].to_numpy()
        if not self.eval:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                self.transform(torch.as_tensor(self.Z_iv[item], dtype=torch.float32)), \
                torch.as_tensor(self.iv_names[item], dtype=torch.int)
        else:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                torch.as_tensor(self.Z_obs[item], dtype=torch.float32)


class ChambersDatasetMultiviewOLD(Dataset):
    def __init__(self, dataset, task, data_root, n_envs=None):
        super().__init__()

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)

        # Select the n_envs
        # if n_envs is not None:
        #     self.env_list = self.env_list[:n_envs]

        chamber_data = ChamberData(name=self.chamber_data_name,
                                   root=self.data_root,
                                   download=True)

        # Observational data
        # obs_data = chamber_data.get_experiment(
        #     name=f'{self.exp}_reference').as_pandas_dataframe()

        # Interventional data
        iv_data_list = [chamber_data.get_experiment(
            name=f'{self.exp}_{env}').as_pandas_dataframe() for env in
                        self.env_list]

        # Enforce that all iv_data have the same length
        n_list = [len(df) for df in iv_data_list]
        n_min = min(n_list)
        iv_data_list = [df[:n_min] for df in iv_data_list]

        # Build views out of iv environments
        n_half = int(n_min/2)
        view_list = []
        for df in iv_data_list:
            view_list.append(df[:n_half])
            view_list.append(df[n_half:])

        self.views_data = view_list

        # Get one big df for all iv data
        # self.iv_data = pd.concat(iv_data_list)

        # Generate intervention index list
        # iv_names = []
        # for idx, iv_data in enumerate(iv_data_list):
        #     iv_names.append(np.repeat(idx, len(iv_data)))
        # self.iv_names = np.concatenate(iv_names)

        # Resample observational data to have same nr of samples as iv_data
        # self.obs_data = obs_data.loc[np.random.choice(len(obs_data),
        #                                               size=len(self.iv_data),
        #                                               replace=True), :]

        # Get subsets of views with shared contents
        self.subsets = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

        # Get indices of shared content variables between views
        self.content_indices = [[0], [1], [2], [3], [4]]

        self.iv_names = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

        if n_envs is not None:
            self.views_data = self.views_data[:(2*n_envs)]
            self.subsets = self.subsets[:n_envs]
            self.content_indices = self.content_indices[:n_envs]
            self.iv_names = self.iv_names[:(2*n_envs)]

    def __len__(self):
        return len(self.views_data[0])

    def __getitem__(self, item):
        """Sample paired data that share an intervened upon dimension"""
        # Get all paths for all views
        view_samples = []
        for i, view in enumerate(self.views_data):
            sample_path = os.path.join(self.data_root, self.chamber_data_name,
                                       _map_iv_envs(self.iv_names[i], self.exp,
                                                    self.env_list),
                                       'images_64',
                                       view['image_file'].iloc[item])
            sample = io.imread(sample_path)
            sample = sample / 255.0
            view_samples.append(torch.as_tensor(sample.transpose((2, 0, 1)),
                                                dtype=torch.float32))

        return view_samples


        # # Interventional sample
        # iv_img_name = os.path.join(self.data_root, self.chamber_data_name,
        #                            _map_iv_envs(self.iv_names[item],
        #                                         self.exp, self.env_list),
        #                            'images_64',
        #                            self.iv_data['image_file'].iloc[item])
        # iv_sample = io.imread(iv_img_name)
        #
        # # Get samples from same interventional environment
        # item_paired = np.random.choice(np.argwhere(self.iv_names == self.iv_names[item]).squeeze())
        #
        # pair_img_name = os.path.join(self.data_root, self.chamber_data_name,
        #                              _map_iv_envs(self.iv_names[item_paired],
        #                                           self.exp, self.env_list),
        #                              'images_64',
        #                              self.iv_data['image_file'].iloc[item_paired])
        # pair_sample = io.imread(pair_img_name)
        #
        # # Normalize
        # iv_sample = iv_sample / 255.0
        # pair_sample = pair_sample / 255.0
        #
        # return torch.as_tensor(iv_sample.transpose((2, 0, 1)), dtype=torch.float32), \
        #     torch.as_tensor(pair_sample.transpose((2, 0, 1)), dtype=torch.float32)


class ChambersDatasetMultiview(Dataset):
    def __init__(self, dataset, task, data_root, include_iv_data=False):
        super().__init__()
        self.eval = False

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)

        assert set(self.env_list).issubset({'red', 'green', 'blue', 'pol_1', 'pol_2'})

        chamber_data = ChamberData(name=self.chamber_data_name,
                                   root=self.data_root,
                                   download=True)
        # Observational data
        obs_data = chamber_data.get_experiment(
            name=f'{self.exp}_reference').as_pandas_dataframe()
        # Interventional data
        iv_data_list = [chamber_data.get_experiment(
            name=f'{self.exp}_{env}').as_pandas_dataframe() for env in
                        self.env_list]
        # Enforce that all iv_data have the same length
        n_list = [len(df) for df in iv_data_list]
        n_min = min(n_list)
        iv_data_list = [df[:n_min] for df in iv_data_list]

        if include_iv_data:
            data_list = [obs_data] + iv_data_list
            self.data = pd.concat(data_list)
            self.env_list = ['reference'] + self.env_list
            self.iv_names = np.concatenate((np.repeat([0], len(obs_data)),
                                            np.repeat(np.arange(1, len(iv_data_list)+1), n_min)))
        else:
            self.data = obs_data
            self.iv_names = np.repeat([0], len(obs_data))
        # Get scalers for standardization
        self.scaler_view2 = StandardScaler()
        self.scaler_view2.fit(self.data[['current', 'ir_1', 'ir_2']].values)

        self.scaler_view3 = StandardScaler()
        self.scaler_view3.fit(self.data[['angle_1']].values)

        self.scaler_view4 = StandardScaler()
        self.scaler_view4.fit(self.data[['angle_2']].values)

        self.subsets = [(0, 1), (0, 2), (0, 3)]
        self.content_indices = [[0, 1, 2], [3], [4]]

        # (Standardized) ground truth factors
        scaler_Z = StandardScaler()
        self.Z = scaler_Z.fit_transform(self.data[self.features].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # View 1: image
        img_path = os.path.join(self.data_root, self.chamber_data_name,
                                _map_iv_envs(self.iv_names[item],
                                             self.exp,
                                             self.env_list),
                                'images_64',
                                self.data['image_file'].iloc[item])
        img_sample = io.imread(img_path)
        img_sample = img_sample / 255.0
        view_1 = torch.as_tensor(img_sample.transpose(2, 0, 1), dtype=torch.float32)

        # View 2: Current, Intensity_1, Intensity_2
        view_2_list = self.data[['current', 'ir_1', 'ir_2']].iloc[item]

        view_2 = torch.as_tensor(self.scaler_view2.transform(np.expand_dims(view_2_list.to_numpy(), 0)),
                                 dtype=torch.float32).squeeze()

        # View 3: Angle 1
        view_3 = torch.as_tensor(self.scaler_view3.transform([[self.data['angle_1'].iloc[item]]]),
                                 dtype=torch.float32).squeeze().unsqueeze(dim=0)

        # View 4: Angle 2
        view_4 = torch.as_tensor(self.scaler_view4.transform([[self.data['angle_2'].iloc[item]]]),
                                 dtype=torch.float32).squeeze().unsqueeze(dim=0)

        if not self.eval:
            return view_1, view_2, view_3, view_4
        else:
            return view_1, view_2, view_3, view_4, \
                torch.as_tensor(self.Z[item], dtype=torch.float32)


class ChambersDatasetMultiviewSynthetic(Dataset):
    def __init__(self, d=5, n=10000, gt_model='model_1'):
        super().__init__()
        self.eval = False

        self.n = n

        self.gt_model = gt_model

        # Sample ground truth latents
        rs = np.random.RandomState(42)
        self.Z = rs.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n)

        match self.gt_model:
            case 'model_1':
                self.subsets = [(0, 1), (0, 2), (0, 3)]
                self.content_indices = [[0, 1, 2], [2, 4], [0, 3]]

                # Sample mixing functions
                self.enc_view_0 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_1 = construct_invertible_mlp(n=3,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_2 = construct_invertible_mlp(n=2,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_3 = construct_invertible_mlp(n=2,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)

                # Apply mixing functions
                self.x_0 = self.enc_view_0(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 3, 4]],
                                    dtype=torch.float32))
                self.x_1 = self.enc_view_1(
                    torch.as_tensor(self.Z[:, [0, 1, 2]],
                                    dtype=torch.float32))
                self.x_2 = self.enc_view_2(
                    torch.as_tensor(self.Z[:, [2, 4]],
                                    dtype=torch.float32))
                self.x_3 = self.enc_view_3(
                    torch.as_tensor(self.Z[:, [0, 3]],
                                    dtype=torch.float32))

                # Standardize latents
                self.x_0 = (self.x_0 - torch.mean(self.x_0, dim=0)) / torch.std(
                    self.x_0, dim=0)
                self.x_1 = (self.x_1 - torch.mean(self.x_1, dim=0)) / torch.std(
                    self.x_1, dim=0)
                self.x_2 = (self.x_2 - torch.mean(self.x_2, dim=0)) / torch.std(
                    self.x_2, dim=0)
                self.x_3 = (self.x_3 - torch.mean(self.x_3, dim=0)) / torch.std(
                    self.x_3, dim=0)

            case 'model_2':
                self.subsets = [(0, 1), (0, 2), (1, 2)]
                self.content_indices = [[0], [1], [2]]

                self.enc_view_0 = construct_invertible_mlp(n=2,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_1 = construct_invertible_mlp(n=2,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_2 = construct_invertible_mlp(n=2,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)

                self.x_0 = self.enc_view_0(
                    torch.as_tensor(self.Z[:, [0, 1]],
                                    dtype=torch.float32))
                self.x_1 = self.enc_view_1(
                    torch.as_tensor(self.Z[:, [0, 2]],
                                    dtype=torch.float32))
                self.x_2 = self.enc_view_2(
                    torch.as_tensor(self.Z[:, [1, 2]],
                                    dtype=torch.float32))

                self.x_0 = (self.x_0 - torch.mean(self.x_0, dim=0)) / torch.std(
                    self.x_0, dim=0)
                self.x_1 = (self.x_1 - torch.mean(self.x_1, dim=0)) / torch.std(
                    self.x_1, dim=0)
                self.x_2 = (self.x_2 - torch.mean(self.x_2, dim=0)) / torch.std(
                    self.x_2, dim=0)

            case 'chamber_synth_indep':
                self.subsets = [(0, 1), (0, 2), (0, 3)]
                self.content_indices = [[0, 1, 2], [3], [4]]

                # Sample mixing functions
                self.enc_view_0 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_1 = construct_invertible_mlp(n=3,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_2 = construct_invertible_mlp(n=1,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_3 = construct_invertible_mlp(n=1,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)

                # Apply mixing functions
                self.x_0 = self.enc_view_0(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 3, 4]],
                                    dtype=torch.float32))
                self.x_1 = self.enc_view_1(
                    torch.as_tensor(self.Z[:, [0, 1, 2]],
                                    dtype=torch.float32))
                self.x_2 = self.enc_view_2(
                    torch.as_tensor(self.Z[:, [3]],
                                    dtype=torch.float32))
                self.x_3 = self.enc_view_3(
                    torch.as_tensor(self.Z[:, [4]],
                                    dtype=torch.float32))

                # Standardize latents
                self.x_0 = (self.x_0 - torch.mean(self.x_0, dim=0)) / torch.std(
                    self.x_0, dim=0)
                self.x_1 = (self.x_1 - torch.mean(self.x_1, dim=0)) / torch.std(
                    self.x_1, dim=0)
                self.x_2 = (self.x_2 - torch.mean(self.x_2, dim=0)) / torch.std(
                    self.x_2, dim=0)
                self.x_3 = (self.x_3 - torch.mean(self.x_3, dim=0)) / torch.std(
                    self.x_3, dim=0)

            case 'chamber_synth_scm':
                # Re-sample gt from SCM
                variances = rs.uniform(0.0, 0.1, size=d)

                W = np.array([
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]
                ])

                lganm = LGANM(W=W, means=np.zeros(d),
                              variances=variances, random_state=42)

                self.Z = lganm.sample(n)


                self.subsets = [(0, 1), (0, 2), (0, 3)]
                self.content_indices = [[0, 1, 2], [3], [4]]

                # Sample mixing functions
                self.enc_view_0 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_1 = construct_invertible_mlp(n=3,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_2 = construct_invertible_mlp(n=1,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_3 = construct_invertible_mlp(n=1,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=25000,
                                                           cond_thresh_ratio=0.001)

                # Apply mixing functions
                self.x_0 = self.enc_view_0(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 3, 4]],
                                    dtype=torch.float32))
                self.x_1 = self.enc_view_1(
                    torch.as_tensor(self.Z[:, [0, 1, 2]],
                                    dtype=torch.float32))
                self.x_2 = self.enc_view_2(
                    torch.as_tensor(self.Z[:, [3]],
                                    dtype=torch.float32))
                self.x_3 = self.enc_view_3(
                    torch.as_tensor(self.Z[:, [4]],
                                    dtype=torch.float32))

                # Standardize latents
                self.x_0 = (self.x_0 - torch.mean(self.x_0, dim=0)) / torch.std(
                    self.x_0, dim=0)
                self.x_1 = (self.x_1 - torch.mean(self.x_1, dim=0)) / torch.std(
                    self.x_1, dim=0)
                self.x_2 = (self.x_2 - torch.mean(self.x_2, dim=0)) / torch.std(
                    self.x_2, dim=0)
                self.x_3 = (self.x_3 - torch.mean(self.x_3, dim=0)) / torch.std(
                    self.x_3, dim=0)

            case 'reprod':
                # Ground truth setting from original paper
                self.subsets = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]
                self.content_indices = [[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 2, 5], [0, 1, 4, 5], [0, 1, 3, 5], [0, 1, 2], [0, 1, 4], [0, 1, 3], [0, 1, 5], [0, 1]]

                self.enc_view_0 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=2500,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_1 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=2500,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_2 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=2500,
                                                           cond_thresh_ratio=0.001)
                self.enc_view_3 = construct_invertible_mlp(n=5,
                                                           n_layers=3,
                                                           n_iter_cond_thresh=2500,
                                                           cond_thresh_ratio=0.001)

                self.x_0 = self.enc_view_0(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 3, 4]],
                                    dtype=torch.float32))
                self.x_1 = self.enc_view_1(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 4, 5]],
                                    dtype=torch.float32))
                self.x_2 = self.enc_view_2(
                    torch.as_tensor(self.Z[:, [0, 1, 2, 3, 5]],
                                    dtype=torch.float32))
                self.x_3 = self.enc_view_3(
                    torch.as_tensor(self.Z[:, [0, 1, 3, 4, 5]],
                                    dtype=torch.float32))

                self.x_0 = (self.x_0 - torch.mean(self.x_0, dim=0)) / torch.std(
                    self.x_0, dim=0)
                self.x_1 = (self.x_1 - torch.mean(self.x_1, dim=0)) / torch.std(
                    self.x_1, dim=0)
                self.x_2 = (self.x_2 - torch.mean(self.x_2, dim=0)) / torch.std(
                    self.x_2, dim=0)
                self.x_3 = (self.x_3 - torch.mean(self.x_3, dim=0)) / torch.std(
                    self.x_3, dim=0)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        if not self.eval:
            return self.x_0[item], self.x_1[item], self.x_2[item], self.x_3[item]
        else:
            return self.x_0[item], self.x_1[item], self.x_2[item], self.x_3[item], self.Z[item]


class ChambersDatasetMultiviewSemisynthetic(ChambersDatasetMultiview):
    def __init__(self, dataset, task, data_root, transform_list, include_iv_data=False):
        super().__init__(dataset, task, data_root, include_iv_data)

        self.transform_view_1, self.transform_view_2, \
            self.transform_view_3, self.transform_view_4 = transform_list

        # Standardize the non-image views
        # View 2:
        z_view_2 = self.data[['red', 'green', 'blue']].values
        self.view_2 = self.transform_view_2(
            torch.as_tensor(z_view_2, dtype=torch.float32))
        self.view_2 = (self.view_2 - torch.mean(self.view_2, dim=0)) / torch.std(self.view_2, dim=0)

        # View 3:
        z_view_3 = self.data['pol_1'].values
        self.view_3 = self.transform_view_3(
            torch.as_tensor(z_view_3, dtype=torch.float32).unsqueeze(dim=-1))
        self.view_3 = (self.view_3 - torch.mean(self.view_3, dim=0)) / torch.std(self.view_3, dim=0)

        # View 4:
        z_view_4 = self.data['pol_2'].values
        self.view_4 = self.transform_view_4(
            torch.as_tensor(z_view_4, dtype=torch.float32).unsqueeze(dim=-1))
        self.view_4 = (self.view_4 - torch.mean(self.view_4, dim=0)) / torch.std(self.view_4, dim=0)

    def __getitem__(self, item):
        # View 1: image
        z_view_1 = self.data[self.features].iloc[item]
        if inspect.ismethod(self.transform_view_1):  # for decoder simulator
            im = self.transform_view_1(z_view_1.to_frame().T).squeeze().transpose(2, 0, 1)
            im = im / 255.0
            view_1 = torch.as_tensor(im, dtype=torch.float32)
        else:
            view_1 = self.transform_view_1(torch.as_tensor(z_view_1, dtype=torch.float32))

        view_2 = self.view_2[item]
        view_3 = self.view_3[item]
        view_4 = self.view_4[item]

        if not self.eval:
            return view_1, view_2, view_3, view_4
        else:
            return view_1, view_2, view_3, view_4, \
                torch.as_tensor(self.Z[item], dtype=torch.float32)

