import os

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, data, data_root):
        self.data = data
        self.data_root = data_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_root, self.data['image_file'].iloc[item])
        sample = io.imread(img_path)

        # Normalize input
        sample = sample / 255.0

        return torch.as_tensor(sample.transpose((2, 0, 1)), dtype=torch.float32)


class PCLEmbeddingDataset(EmbeddingDataset):
    def __getitem__(self, item):
        if item == 0:
            item += 1
        rand_item = np.random.choice(len(self.data))
        
        sample = super().__getitem__(item)
        sample_tm1 = super().__getitem__(item - 1)
        sample_rand = super().__getitem__(rand_item)

        X = torch.stack((sample, sample_tm1), dim=0)
        X_perm = torch.stack((sample, sample_rand), dim=0)

        return torch.stack((X, X_perm))
