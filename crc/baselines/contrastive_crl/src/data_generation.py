import sempler
from sempler.generators import dag_avg_deg
from sempler.lganm import _parse_interventions
import numpy as np
import os
import pandas as pd
import random
import torch

from skimage import io
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset as TensorDataset
import torchvision.transforms as transforms

from crc.baselines.contrastive_crl.src.nonlinearities import Identity, Linear_Nonlinearity, ImageGenerator
from crc.baselines.contrastive_crl.src.models import EmbeddingNet
from crc.utils import get_task_environments

from causalchamber.datasets import Dataset as ChamberData


def get_data_from_kwargs(data_kwargs):
    function_type = data_kwargs['mixing']
    torch.manual_seed(data_kwargs['seed'])
    if function_type == 'identity':
        f = Identity()
        data_kwargs['dim_x'] = data_kwargs['d']
    elif function_type == 'linear':
        f = Linear_Nonlinearity(data_kwargs['d'], data_kwargs['dim_x'])
    elif function_type == 'mlp':
        f = EmbeddingNet(data_kwargs['d'], data_kwargs['dim_x'], data_kwargs['hidden_dim'],
                         hidden_layers=data_kwargs['hidden_layers'], residual=False)
    elif function_type == 'image':
        f = ImageGenerator()
    data_kwargs['f'] = f
    return DataBag(**data_kwargs)


class DataBag:
    def __init__(self, d, k, n, f=None, var_range_obs=(1., 1.), var_range_int=(1., 1.), mean_range=(0., 0.),
                 var_shift=True, seed=0, train_fraction=.8, repeat_obs_samples=False, noise_type='gaussian',
                 normalize=False, mixing='Identity', **kwargs):
        np.random.seed(seed)
        random.seed(seed)
        self.mixing = mixing
        self.d = d
        self.k = min(d, k)
        self.n = n
        self.f = f if f is not None else Identity()
        self.cache = True if d * n < 150000 else False
        self.scale_factors = np.ones((1,d))
        self.normalize = normalize
        self.generator = np.random.default_rng(seed)
        self.repeat_obs_samples = repeat_obs_samples
        self.train_samples = int(n * train_fraction)
        self.val_samples = n - self.train_samples
        self.test_samples = 5000
        self.noise_type = noise_type
        self.constrain_samples = kwargs.get('constrain_to_image', False)
        W, self.ordering = dag_avg_deg(p=d, k=k, w_min=.25, w_max=1., return_ordering=True, random_state=seed)
        self.W = self.add_signs(W)
        self.var_obs = np.random.uniform(var_range_obs[0], var_range_obs[1], size=d)
        self.var_int = np.random.uniform(var_range_int[0], var_range_int[1], size=d)
        if not var_shift:
            self.var_int = self.var_obs
        self.mean_int = self.add_signs(np.random.uniform(mean_range[0], mean_range[1], size=d))
        self.lganm = sempler.LGANM(self.W, np.zeros(d), self.var_obs)
        self.obs, self.int, self.targets = self.sample(self.train_samples, repeat_obs_samples)
        self.obs_val, self.int_val, self.targets_val = self.sample(self.val_samples, repeat_obs_samples)
        obs_test, int_test, targets_test = self.sample(self.test_samples,
                                                       repeat_obs_samples=False)
        self.obs_test = obs_test[:self.test_samples]
        self.int_test = int_test[:self.test_samples]
        self.targets_test = targets_test[:self.test_samples]
        self.obs_f, self.int_f, self.obs_f_val, self.int_f_val = self.get_observations_cached()

    @staticmethod
    def add_signs(W):
        mask = np.random.binomial(n=1, p=.5, size=W.size).reshape(W.shape)
        return W - 2 * mask * W

    def sample_from_dag(self, n, do_interventions={}):
        if not self.constrain_samples:
            return self.sample_lganm_copy(n, do_interventions=do_interventions)
        samples = self.sample_lganm_copy(3 * n, do_interventions=do_interventions)
        mask = np.all(np.abs(samples) < .5, axis=1)
        selected_rows = samples[mask]
        print('{:.1%} of the samples were discarded because constraint was not satisfied'.format(
            1 - selected_rows.shape[0] / (3 * n))
        )
        if selected_rows.shape[0] < n:
            raise ValueError('Not enough samples created')
        return selected_rows[:n]

    def get_observations(self, samples=None, environment=0, validation=True):
        if samples is None:
            samples = self.val_samples if validation else self.train_samples
        if environment == 0:
            if validation:
                return self.f(torch.tensor(self.obs_val[:samples], dtype=torch.float)).detach().numpy()
            else:
                return self.f(torch.tensor(self.obs[:samples], dtype=torch.float)).detach().numpy()
        if validation:
            start_index = np.where(self.targets_val == environment-1)[0][0]
            samples = min(samples, self.val_samples)
            return self.f(torch.tensor(self.int_val[start_index:(start_index+samples)], dtype=torch.float)).detach().numpy()
        else:
            start_index = np.where(self.targets == environment-1)[0][0]
            samples = min(samples, self.train_samples)
            return self.f(torch.tensor(self.int[start_index:(start_index+samples)], dtype=torch.float)).detach().numpy()

    def get_observations_cached(self):
        if not self.cache:
            return None, None, None, None
        obs_f = self.f(torch.tensor(self.obs, dtype=torch.float)).detach().numpy()
        obs_val_f = self.f(torch.tensor(self.obs_val, dtype=torch.float)).detach().numpy()
        int_f = self.f(torch.tensor(self.int, dtype=torch.float)).detach().numpy()
        int_val_f = self.f(torch.tensor(self.int_val, dtype=torch.float)).detach().numpy()
        return obs_f, int_f, obs_val_f, int_val_f

    def sample(self, n, repeat_obs_samples=False):
        if repeat_obs_samples:
            obs_data = self.sample_from_dag(n)
            if self.normalize:
                self.scale_factors = np.std(obs_data, axis=0, keepdims=True) #  / np.std(obs_data)
            obs_data = obs_data / self.scale_factors
            obs_data = np.tile(obs_data, (self.d, 1))
            # np.random.shuffle(obs_data)
        else:
            obs_data = self.sample_from_dag(n * self.d)
            if self.normalize:
                self.scale_factors = np.std(obs_data, axis=0, keepdims=True) # / np.std(obs_data)
            obs_data = obs_data / self.scale_factors
        int_data = []
        target = []
        for i in range(self.d):
            samples = self.sample_from_dag(n, do_interventions={i: (self.mean_int[i], self.var_int[i])})
            samples = samples / self.scale_factors
            int_data.append(samples)
            target.append(i * np.ones(n, dtype=np.int_))
        int_data = np.concatenate(int_data, axis=0)
        target = np.concatenate(target, axis=0)
        return obs_data, int_data, target

    def get_dataloaders(self, batch_size, train=True):
        if self.cache:
            return self.get_dataloaders_cached(batch_size, train)
        obs = self.obs if train else self.obs_val
        intven = self.int if train else self.int_val
        targets = self.targets if train else self.targets_val
        obs_dataset = ObservationalDataset(obs, self.f)
        int_dataset = InterventionalDataset(intven, self.f, targets)
        obs_dataloader = DataLoader(obs_dataset, shuffle=True, batch_size=batch_size)
        int_dataloader = DataLoader(int_dataset, shuffle=True, batch_size=batch_size)
        return obs_dataloader, int_dataloader

    def get_datasets(self, mode='train'):
        if mode == 'train':
            obs = self.obs
            intven = self.int
            targets = self.targets
        elif mode == 'val':
            obs = self.obs_val
            intven = self.int_val
            targets = self.targets_val
        elif mode == 'test':
            obs = self.obs_test
            intven = self.int_test
            targets = self.targets_test
        else:
            raise ValueError("Invalid mode passed! Must be train, val or test.")

        dataset = ContrastiveCRLDataset(obs, intven, self.f, targets, self.W)
        return dataset

    def get_dataloaders_cached(self, batch_size, train=True):
        obs_f = self.obs_f if train else self.obs_f_val
        int_f = self.int_f if train else self.int_f_val
        targets = self.targets if train else self.targets_val
        obs_dataloader = DataLoader(torch.tensor(obs_f, dtype=torch.float), shuffle=True,
                                    batch_size=batch_size)
        int_dataloader = DataLoader(TensorDataset(torch.tensor(int_f, dtype=torch.float), torch.tensor(targets)),
                                    shuffle=True, batch_size=batch_size)
        return obs_dataloader, int_dataloader


    def get_dataset_for_linear_disentanglement(self):
        B_obs = (1 / np.sqrt(self.var_obs)).reshape(-1, 1) * (np.eye(self.d) - self.W.T)
        P = np.eye(self.d)
        H = self.get_H()
        Bs = self.d * [B_obs]
        ix2target = dict()
        for i in range(self.d):
            ix2target[i] = i

        covariance = np.cov(self.obs_f, rowvar=False)
        Theta = np.linalg.pinv(covariance)
        Theta = self.project_on_top_eigenspaces(Theta, self.d)
        Thetas = []
        for i in range(self.d):
            covariance = np.cov(self.int_f[self.targets == i], rowvar=False)
            precision = np.linalg.pinv(covariance)
            precision = self.project_on_top_eigenspaces(precision, self.d)
            Thetas.append(precision)
        return (B_obs, P, H, Bs, ix2target), (Theta, Thetas)

    @staticmethod
    def project_on_top_eigenspaces(x, dim):
        u, s, vh = np.linalg.svd(x, hermitian=True, full_matrices=True)
        return (u[:, :dim] * s[:dim]) @ vh[:dim, :]


    def get_H(self):
        obs_copy = np.copy(self.obs)
        obs_f_copy = np.copy(self.obs_f)
        cov = np.cov(obs_f_copy, rowvar=False)
        cov = self.project_on_top_eigenspaces(cov, self.d)
        obs_copy = np.expand_dims(obs_copy, 1)
        obs_f_copy = np.expand_dims(obs_f_copy, 2)
        correlations = np.mean(obs_copy * obs_f_copy, axis=0)
        return (np.linalg.pinv(cov) @ correlations).T

    def sample_lganm_copy(self, n=100, do_interventions={}):
        """
        This is essentially a copy of the sampling function from the sempler package that in addition can sample using
        different noise distribution. For future usage a different sampling package should be used. We only copy the
        do_interventions because only they are used.
        :param n:
        :param population:
        :param do_interventions:
        :param shift_interventions:
        :param noise_interventions:
        :param random_state:
        :return:
        """
        W = self.W.copy()
        variances = self.lganm.variances.copy()
        means = self.lganm.means.copy()
        if do_interventions:
            do_interventions = _parse_interventions(do_interventions)
            targets = do_interventions[:, 0].astype(int)
            means[targets] = do_interventions[:, 1]
            variances[targets] = do_interventions[:, 2]
            W[:, targets] = 0

        # Sampling by building the joint distribution
        A = np.linalg.inv(np.eye(self.lganm.p) - W.T)

        noise_variables = self.sample_noise_variables(n)
        std_reshape = np.reshape(np.sqrt(variances), (1, self.d))
        mean_reshape = np.reshape(means, (1, self.d))
        noise_variables = std_reshape * noise_variables + mean_reshape
        return (A @ noise_variables.T).T

    def sample_noise_variables(self, n):
        if self.noise_type == "laplace":
            return self.generator.laplace(0, 1 / np.sqrt(2), (n, self.d))
        elif self.noise_type == "exponential":
            return self.generator.exponential(1, (n, self.d)) - 1
        elif self.noise_type == "gaussian":
            return self.generator.normal(0, 1, (n, self.d))
        elif self.noise_type == "gumbel":
            scale = np.sqrt(6) / np.pi
            loc = - 0.5772 * scale
            return self.generator.gumbel(loc, scale, (n, self.d))
        elif self.noise_type == "uniform":
            return self.generator.uniform(-np.sqrt(3), np.sqrt(3), (n, self.d))
        else:
            raise NotImplementedError("Noise type {} is not implemented".format(self.noise_type))


class ObservationalDataset(Dataset):
    def __init__(self, z, f):
        self.z = torch.tensor(z, dtype=torch.float)
        self.f = f
        self.transform = transforms.Compose([f])

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.transform(self.z[idx])


class InterventionalDataset(Dataset):
    def __init__(self, z, f, t):
        self.z = torch.tensor(z, dtype=torch.float)
        self.f = f
        self.t = torch.tensor(t)
        self.transform = transforms.Compose([f])

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        # Return a tuple (sample, target) at the given index
        return self.transform(self.z[idx]), self.t[idx]


class ContrastiveCRLDataset(Dataset):
    def __init__(self, z_obs, z_int, f, t, W):
        self.z_obs = torch.tensor(z_obs, dtype=torch.float)
        self.z_int = torch.tensor(z_int, dtype=torch.float)
        self.f = f
        self.t = torch.tensor(t, dtype=torch.int)
        self.W = W
        self.transform = transforms.Compose([f])

    def __len__(self):
        return len(self.z_obs)

    def __getitem__(self, idx):
        # Return a tuple (obs_sample, int_sample, target) at the given index
        return self.transform(self.z_obs[idx]), self.transform(self.z_int[idx]), self.t[idx]


class ChamberDataset(Dataset):
    def __init__(self,
                 dataset,
                 task,
                 data_root,
                 eval=False,
                 transform=None):
        super(Dataset, self).__init__()
        self.eval = eval

        self.transform = transform

        self.data_root = data_root
        self.chamber_data_name = dataset
        self.exp, self.env_list = get_task_environments(task)
        chamber_data = ChamberData(self.chamber_data_name, root=self.data_root,
                                   download=True)

        # Observational data
        obs_data = chamber_data.get_experiment(
            name=f'{self.exp}_reference').as_pandas_dataframe()
        # Interventional data
        iv_data_1 = chamber_data.get_experiment(name=f'{self.exp}_red').as_pandas_dataframe()
        iv_data_2 = chamber_data.get_experiment(name=f'{self.exp}_green').as_pandas_dataframe()
        iv_data_3 = chamber_data.get_experiment(name=f'{self.exp}_blue').as_pandas_dataframe()
        iv_data_4 = chamber_data.get_experiment(name=f'{self.exp}_pol_1').as_pandas_dataframe()
        iv_data_5 = chamber_data.get_experiment(name=f'{self.exp}_pol_2').as_pandas_dataframe()
        iv_data_list = [iv_data_1, iv_data_2, iv_data_3, iv_data_4, iv_data_5]
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
        if self.exp in ('scm_1', 'scm_2'):
            # TODO: probably need to follow some convention of making this upper triang
            self.W = np.array(
                [
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                ]
            )

    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        # Observational sample
        obs_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                    f'{self.exp}_reference',
                                    'images_64',
                                    self.obs_data['image_file'].iloc[item])
        obs_sample = io.imread(obs_img_name)
        # Interventional sample
        iv_img_name = os.path.join(self.data_root, self.chamber_data_name,
                                   self._map_iv_envs(self.iv_names[item], self.exp),
                                   'images_64',
                                   self.iv_data['image_file'].iloc[item])
        iv_sample = io.imread(iv_img_name)

        # Normalize inputs
        obs_sample = obs_sample / 255.0
        iv_sample = iv_sample / 255.0

        if not self.eval:
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32), \
                torch.as_tensor(iv_sample.transpose((2, 0, 1)),
                                dtype=torch.float32), \
                torch.as_tensor(self.iv_names[item],
                                dtype=torch.int)
        else: # also return the ground truth variables
            Z_obs = self.obs_data[['red', 'green', 'blue', 'pol_1', 'pol_2']].iloc[item].to_numpy()
            Z_iv = self.iv_data[['red', 'green', 'blue', 'pol_1', 'pol_2']].iloc[item].to_numpy()
            return torch.as_tensor(obs_sample.transpose((2, 0, 1)),
                                   dtype=torch.float32), \
                torch.as_tensor(iv_sample.transpose((2, 0, 1)),
                                dtype=torch.float32), \
                torch.as_tensor(self.iv_names[item],
                                dtype=torch.int), \
                Z_obs, Z_iv

    @staticmethod
    def _map_iv_envs(idx, exp):
        # idx = int(idx)
        map = [f'{exp}_red', f'{exp}_green', f'{exp}_blue', f'{exp}_pol_1', f'{exp}_pol_2']

        return map[idx]
