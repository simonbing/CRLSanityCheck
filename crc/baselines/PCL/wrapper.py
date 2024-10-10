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
                                       task=self.task, whiten=False)

        dl_train = DataLoader(dataset_train, shuffle=True, batch_size=self.batch_size)

        # Save train data
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path) or self.overwrite_data:
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        best_model_path = os.path.join(self.train_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            # Training (only runs if saved model doesn't exist yet)
            train(data=dl_train,
                  epochs=self.epochs,
                  random_seed=self.seed,
                  list_hidden_nodes=[128, 128] + [self.lat_dim],
                  initial_learning_rate=0.1,
                  momentum=0.9,
                  max_steps=None,
                  decay_steps=max(1, int(self.epochs / 2)),
                  decay_factor=0.1,
                  batch_size=self.batch_size,
                  train_dir=self.train_dir,
                  in_dim=64*64*3,  # hardcoded for image data
                  latent_dim=self.lat_dim,
                  ar_order=1,
                  weight_decay=1e-5,
                  checkpoint_steps=2 * self.epochs,
                  moving_average_decay=0.999,
                  summary_steps=max(1, int(self.epochs / 10)))


class EvalPCL(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        pass
