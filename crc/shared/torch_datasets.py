import inspect
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
from crc.baselines.contrastive_crl.src.models import EmbeddingNet


def _map_iv_envs(idx, exp, env_list):
    map = [f'{exp}_{env}' for env in env_list]

    return map[idx]


class ChambersDatasetContrastive(Dataset):
    def __init__(self, dataset, task, data_root):
        """
        Parameters
        ----------
            dataset : str
                Name of the chambers dataset.
            task : str
                Experimental task.
            data_root : str
                Path to saved data.
        """
        super().__init__()
        self.eval = False

        self.W = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.71943922, 0.0, 0.0, 0.0, 0.67726298],
            [0.0, 0.89303215, 0.0, 0.0, 0.98534901],
            [0.84868401, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)

        chamber_data = ChamberData(name=self.chamber_data_name,
                                   root=self.data_root,
                                   download=True)

        # Observational data
        obs_data = chamber_data.get_experiment(
            name=f'{self.exp}_obs').as_pandas_dataframe()
        Z_obs = obs_data[self.features].to_numpy()
        # Normalize
        self.means = np.mean(Z_obs, axis=0, keepdims=True)
        self.scale_factors = np.std(Z_obs, axis=0, keepdims=True)
        Z_obs = (Z_obs - self.means) / self.scale_factors
        self.Z_obs = np.tile(Z_obs, (Z_obs.shape[-1], 1))

        self.obs_data = pd.concat([obs_data] * 5)

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

    def __len__(self):
        return len(self.Z_obs)

    def __getitem__(self, item):
        # Observational sample
        obs_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                    f'{self.exp}_obs',
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


class ChambersDatasetContrastiveSemiSynthetic(ChambersDatasetContrastive):
    def __init__(self, dataset, task, data_root, transform):
        """
        Parameters
        ----------
            dataset : str
                Name of the chambers dataset.
            task : str
                Experimental task.
            data_root : str
                Path to saved data.
            transform : callable
                Function that transforms ground-truth values into observations.
        """
        super().__init__(dataset, task, data_root)

        try:
            for p in transform.parameters():
                p.requires_grad = False
        except AttributeError:
            pass

        self.transform = transform

        self.Z_obs_df = self.obs_data
        self.Z_iv_df = self.iv_data

    def __getitem__(self, item):
        if inspect.ismethod(self.transform):  # for decoder simulator
            # We use the non-normalized data as input here
            obs_df = self.Z_obs_df.iloc[item].to_frame().T
            obs_sample = self.transform(obs_df)
            obs_sample = obs_sample / 255.0
            if not self.eval:
                iv_df = self.Z_iv_df.iloc[item].to_frame().T
                iv_sample = self.transform(iv_df)
                iv_sample = iv_sample / 255.0
                return torch.as_tensor(obs_sample.squeeze().transpose(2, 0, 1), dtype=torch.float32), \
                    torch.as_tensor(iv_sample.squeeze().transpose(2, 0, 1), dtype=torch.float32), \
                    torch.as_tensor(self.iv_names[item], dtype=torch.int)
            else:
                return torch.as_tensor(obs_sample.squeeze().transpose(2, 0, 1), dtype=torch.float32), \
                    self.Z_obs[item]
        else:
            if not self.eval:
                return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                    self.transform(torch.as_tensor(self.Z_iv[item], dtype=torch.float32)), \
                    torch.as_tensor(self.iv_names[item], dtype=torch.int)
            else:
                return self.transform(torch.as_tensor(self.Z_obs[item], dtype=torch.float32)), \
                    self.Z_obs[item]


class ChambersDatasetContrastiveSynthetic(Dataset):
    def __init__(self, d, k, n=100000):
        """
        Parameters
        ----------
            d : int
                Ground-truth dimension.
            k : int
                Connectivity of ground-truth graph.
            n : int
                Number of samples.
        """
        super().__init__()
        self.eval = False

        # Sample adjacency
        W, ordering = dag_avg_deg(p=d, k=k, w_min=0.25, w_max=1.0,
                                  return_ordering=True, random_state=42)
        # Add signs
        rs = np.random.RandomState(42)
        mask = rs.binomial(n=1, p=0.5, size=W.size,).reshape(W.shape)
        self.W = W - 2 * mask * W

        variances_obs = rs.uniform(1.0, 2.0, size=d)
        # variances_obs = rs.uniform(0.01, 0.02, size=d)

        lganm = LGANM(W=self.W, means=np.zeros(d), variances=variances_obs, random_state=42)

        # Observational samples
        obs_samples = lganm.sample(n)
        self.means = np.mean(obs_samples, axis=0, keepdims=True)
        self.stds = np.std(obs_samples, axis=0, keepdims=True)

        obs_samples = (obs_samples - self.means) / self.stds

        self.Z_obs = np.tile(obs_samples, (d, 1))

        # Interventional samples
        means_iv = rs.uniform(1.0, 2.0, size=d)
        mask = rs.binomial(n=1, p=0.5, size=means_iv.size).reshape(means_iv.shape)
        means_iv = means_iv - 2 * mask * means_iv
        variances_iv = rs.uniform(1.0, 2.0, size=d)

        iv_samples = []
        iv_targets = []
        for i in range(d):
            samples = lganm.sample(n, do_interventions={i: (means_iv[i], variances_iv[i])})
            samples = (samples - self.means) / self.stds
            iv_samples.append(samples)
            iv_targets.append(i * np.ones(n, dtype=np.int_))
        self.Z_iv = np.concatenate(iv_samples, axis=0)

        self.iv_names = np.concatenate(iv_targets, axis=0)

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



