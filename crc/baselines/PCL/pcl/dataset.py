import os

from causalchamber.datasets import Dataset as ChamberData
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset

from crc.utils import get_task_environments
from crc.baselines.PCL.subfunc.generate_artificial_data_art import \
    generate_artificial_data, apply_mlp_to_source


class ChamberDataset(Dataset):
    def __init__(self, dataset, task, data_root, whiten=False):
        super().__init__()
        self.dataset = dataset
        if self.dataset.endswith('synth_mix'):
            self.synth_mixing = True
            self.dataset = self.dataset.removesuffix('_synth_mix')
        else:
            self.synth_mixing = False

        if self.dataset == 'synth_pcl':
            x, s, _, _, _, _ = generate_artificial_data(num_comp=4,
                                                        num_data=2**16,
                                                        ar_coef= [0.9]*4,
                                                        ar_order=1,
                                                        num_layer=3,
                                                        random_seed=0)
            pca = PCA(whiten=True)
            x = pca.fit_transform(x)
            self.data = x
            self.s = s
        else:
            self.data_root = data_root

            self.exp, self.env_list, self.features = get_task_environments(task)

            chamber_data = ChamberData(self.dataset, root=self.data_root, download=True)
            # PCL only takes a single environment, so only take the first element in env_list
            self.env = self.env_list[0]
            self.data = chamber_data.get_experiment(name=f'{self.exp}_{self.env}').as_pandas_dataframe()

            if self.synth_mixing:
                self.s = self.data[self.features].to_numpy()
                self.s = (self.s - np.mean(self.s, axis=0, keepdims=True)) / \
                         np.std(self.s, axis=0, keepdims=True)
                x, _ = apply_mlp_to_source(self.s, num_layer=3, random_seed=0)

                pca = PCA(whiten=True)
                x = pca.fit_transform(x)
                self.data = x
            else:
                self.whiten = whiten
                if self.whiten:
                    # Whitening
                    self.pca = PCA(whiten=True)
                    img_arr = np.asarray([io.imread(os.path.join(self.data_root,
                                                                 self.dataset,
                                                                 f'{self.exp}_{self.env}',
                                                                 'images_64',
                                                                 self.data['image_file'].iloc[i])).flatten()
                                          for i in range(len(self.data))])
                    self.pca.fit(img_arr)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if item == 0:
            item += 1  # need t-1 time step for each sample

        rand_item = np.random.choice(len(self.data))

        if self.dataset == 'synth_pcl' or self.synth_mixing:
            sample = self.data[item, :]
            tm1_sample = self.data[item - 1, :]
            rand_sample = self.data[rand_item, :]

            X = torch.as_tensor(np.stack((sample, tm1_sample), axis=0),
                                dtype=torch.float32)
            X_perm = torch.as_tensor(np.stack((sample, rand_sample), axis=0),
                                     dtype=torch.float32)
            Z = self.s[item, :]
        else:
            img_path = os.path.join(self.data_root, self.dataset,
                                    f'{self.exp}_{self.env}', 'images_64',
                                    self.data['image_file'].iloc[item])
            img_sample = io.imread(img_path)

            img_tm1_path = os.path.join(self.data_root, self.dataset,
                                        f'{self.exp}_{self.env}', 'images_64',
                                        self.data['image_file'].iloc[item - 1])
            img_tm1_sample = io.imread(img_tm1_path)

            img_rand_path = os.path.join(self.data_root, self.dataset,
                                         f'{self.exp}_{self.env}', 'images_64',
                                         self.data['image_file'].iloc[rand_item])
            img_rand_sample = io.imread(img_rand_path)

            # Normalize inputs
            img_sample = img_sample / 255.0
            img_tm1_sample = img_tm1_sample / 255.0
            img_rand_sample = img_rand_sample / 255.0

            if self.whiten:
                img_sample = self.pca.transform(img_sample)
                img_tm1_sample = self.pca.transform(img_tm1_sample)
                img_rand_sample = self.pca.transform(img_rand_sample)

            X = torch.as_tensor(np.stack((np.transpose(img_sample, (2, 0, 1)),
                                          np.transpose(img_tm1_sample, (2, 0, 1))),
                                         axis=0),
                                dtype=torch.float32)
            X_perm = torch.as_tensor(np.stack((np.transpose(img_sample, (2, 0, 1)),
                                               np.transpose(img_rand_sample, (2, 0, 1))),
                                              axis=0),
                                     dtype=torch.float32)

            Z = self.data[self.features].iloc[item].to_numpy()

        return X, X_perm, \
            torch.ones(1, dtype=torch.float32), \
            torch.zeros(1, dtype=torch.float32), \
            Z

    def __len__(self):
        return len(self.data)
