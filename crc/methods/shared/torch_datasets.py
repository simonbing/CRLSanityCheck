import os

from causalchamber.datasets import Dataset as ChamberData
import numpy as np
import pandas as pd
from sempler.generators import dag_avg_deg
from sempler import LGANM
from skimage import io
import torch
from torch.utils.data import Dataset

from crc.utils import get_task_environments
from crc.methods.shared.architectures import FCEncoder


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

        # Generate intervention index list
        iv_names = []
        for idx, iv_data in enumerate(iv_data_list):
            iv_names.append(np.repeat(idx, len(iv_data)))
        self.iv_names = np.concatenate(iv_names)

        # Resample observational data to have same nr of samples as iv_data
        self.obs_data = obs_data.loc[np.random.choice(len(obs_data),
                                                      size=len(self.iv_data),
                                                      replace=True), :]

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
        W = dag_avg_deg(p=d, k=k, w_min=0.25, w_max=1.0, random_state=42)
        # Add signs
        rs = np.random.RandomState(42)
        mask = rs.binomial(n=1, p=0.5, size=W.size,).reshape(W.shape)
        self.W = W - 2 * mask * W

        variances_obs = rs.uniform(1.0, 2.0, size=d)

        lganm = LGANM(W=self.W, means=np.zeros(d), variances=variances_obs, random_state=42)

        # Observational samples
        obs_samples = lganm.sample(n)
        self.Z_obs = np.tile(obs_samples, (d, 1))

        # Interventional samples
        means_iv = np.random.uniform(1.0, 2.0, size=d)
        mask = rs.binomial(n=1, p=0.5, size=means_iv.size).reshape(means_iv.shape)
        means_iv = means_iv - 2 * mask * means_iv
        variances_iv = rs.uniform(1.0, 2.0, size=d)

        iv_samples = []
        iv_targets = []
        for i in range(d):
            samples = lganm.sample(n, do_interventions={i: (means_iv[i], variances_iv[i])})
            iv_samples.append(samples)
            iv_targets.append(i * np.ones(n, dtype=np.int_))
        self.Z_iv = np.concatenate(iv_samples, axis=0)
        self.iv_targets = np.concatenate(iv_targets, axis=0)

        self.transform = FCEncoder(in_dim=d, latent_dim=x_dim,
                                   hidden_dims=[512, 512, 512], residual=False)

    def __len__(self):
        return len(self.Z_obs)

    def __getitem__(self, item):
        if not self.eval:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                self.transform(torch.as_tensor(self.Z_iv[item], dtype=torch.float32)), \
                torch.as_tensor(self.iv_targets[item], dtype=torch.int)
        else:
            return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                self.Z_obs[item]


class ChambersDatasetContrastiveSemiSynthetic(ChambersDatasetContrastive):
    def __init__(self, dataset, task, data_root, transform):
        super().__init__(dataset, task, data_root)

        self.transform = transform

    def __getitem__(self, item):
        Z_obs = self.obs_data[self.features].iloc[item].to_numpy()
        Z_iv = self.iv_data[self.features].iloc[item].to_numpy()
        if not self.eval:
            return self.transform(torch.as_tensor(Z_obs, dtype=torch.float32)), \
                self.transform(torch.as_tensor(Z_iv, dtype=torch.float32)), \
                torch.as_tensor(self.iv_names[item], dtype=torch.int)
        else:
            return self.transform(Z_obs), Z_obs


class ChambersDatasetMultiview(Dataset):
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
