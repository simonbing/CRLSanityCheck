import os

from skimage import io
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
