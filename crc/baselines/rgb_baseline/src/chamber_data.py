import os

import pandas as pd
import torch
from causalchamber.datasets import Dataset as ChamberData
from skimage import io
from torch.utils.data import Dataset

from crc.utils import get_task_environments


class ChamberDataset(Dataset):
    def __init__(self, dataset, task, data_root):
        super().__init__()

        self.dataset = dataset
        self.exp, self.env_list, self.features = get_task_environments(task)
        self.env_list = ['reference'] + self.env_list
        self.data_root = data_root

        chamber_data = ChamberData(self.dataset, root=self.data_root,
                                   download=True)
        data_list = [chamber_data.get_experiment(
            name=f'{self.exp}_{env}').as_pandas_dataframe() for env in
                     self.env_list]

        for data, env in zip(data_list, self.env_list):
            data.insert(0, 'env_name', env)

        self.data = pd.concat(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample_path = os.path.join(self.data_root, self.dataset,
                                   f"{self.exp}_{self.data['env_name'].iloc[item]}",
                                   'images_64',
                                   self.data['image_file'].iloc[item])

        sample = io.imread(sample_path)

        # Normalize
        sample = sample / 255.0

        Z = self.data[self.features].iloc[item].to_numpy()

        return torch.as_tensor(sample.transpose((2, 0, 1)), dtype=torch.float32), \
            Z
