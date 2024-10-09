import os

from causalchamber.datasets import Dataset as ChamberDataset
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

from crc.utils import get_task_environments


class ChamberDataset(Dataset):
    def __init__(self, dataset, task, data_root, synth_mixing=False):
        super().__init__()
        self.dataset = dataset
        self.data_root = data_root

        self.exp, self.env_list = get_task_environments(task)

        self.features = ['red', 'green', 'blue', 'pol_1', 'pol_2']  # hardcoded for now

        chamber_data = ChamberDataset(self.dataset, root=self.data_root, download=True)
        self.data = chamber_data.get_experiment(name=f'{self.exp}_reference').as_pandas_dataframe()

        self.synth_mixing = synth_mixing
        # Apply synthetic mixing to the data once here and then just get data later!

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if item == 0:
            item += 1  # need t-1 time step for each sample\

        rand_item = np.random.choice(len(self.data))

        if self.synth_mixing:
            pass
        else:
            img_path = os.path.join(self.data_root, self.dataset,
                                    f'{self.exp}_reference', 'images_64',
                                    self.data['image_file'].iloc[item])
            img_sample = io.imread(img_path)

            img_tm1_path = os.path.join(self.data_root, self.dataset,
                                        f'{self.exp}_reference', 'images_64',
                                        self.data['image_file'].iloc[item - 1])
            img_tm1_sample = io.imread(img_tm1_path)

            img_rand_path = os.path.join(self.data_root, self.dataset,
                                         f'{self.exp}_reference', 'images_64',
                                         self.data['image_file'].iloc[rand_item])
            img_rand_sample = io.imread(img_rand_path)

            X = torch.as_tensor(np.concatenate((img_sample.flatten(),
                                                img_tm1_sample.flatten()),
                                               axis=0),
                                dtype=torch.float32)
            X_perm = torch.as_tensor(np.concatenate((img_sample.flatten(),
                                                     img_rand_sample.flatten()),
                                                    axis=0),
                                     dtype=torch.float32)

        return X, X_perm, torch.ones(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
