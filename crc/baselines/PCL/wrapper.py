import os
import pickle

from torch.utils.data import DataLoader

from crc.wrappers import TrainModel, EvalModel
from crc.baselines.PCL.pcl.dataset import ChamberDataset
from crc.baselines.PCL.pcl.train import train


class TrainPCL(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Get data
        dataset_train = ChamberDataset(dataset=self.dataset, data_root=self.data_root,
                                       task=self.task)

        dl_train = DataLoader(dataset_train, shuffle=True, batch_size=self.batch_size)

        # Save train data
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path) or self.overwrite_data:
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Training
        train(data=None,
              random_seed=self.seed,
              list_hidden_nodes=None,
              initial_learning_rate=None,
              momentum=None,
              max_steps=None,
              decay_steps=None,
              decay_factor=None,
              batch_size=self.batch_size,
              train_dir=None,
              latent_dim=None,
              ar_order=1,
              weight_decay=None,
              checkpoint_steps=None,
              moving_average_decay=None,
              summary_steps=None)


class EvalPCL(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        pass
