import os
import pickle

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from crc.wrappers import TrainModel, EvalModel
from crc.baselines.PCL.pcl.dataset import ChamberDataset
from crc.baselines.PCL.pcl.train import train


class TrainPCL(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Get data
        chamber_dataset = ChamberDataset(dataset=self.dataset, data_root=self.data_root,
                                         task=self.task, whiten=False)
        train_frac = 0.8
        train_idx = int(len(chamber_dataset) * train_frac)
        dataset_train = Subset(chamber_dataset, np.arange(len(chamber_dataset))[:train_idx])
        dataset_test = Subset(chamber_dataset, np.arange(len(chamber_dataset))[train_idx:])

        # Save train data
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path) or self.overwrite_data:
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save test data
        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path) or self.overwrite_data:
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataset_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        dl_train = DataLoader(dataset_train, shuffle=True,
                              batch_size=self.batch_size)

        best_model_path = os.path.join(self.train_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            # Training (only runs if saved model doesn't exist yet)
            train(data=dl_train,
                  epochs=self.epochs,
                  random_seed=self.seed,
                  list_hidden_nodes=[8, 8] + [self.lat_dim],
                  initial_learning_rate=0.1,
                  momentum=0.9,
                  max_steps=None,
                  decay_steps=max(1, int(self.epochs / 2)),
                  decay_factor=0.1,
                  batch_size=self.batch_size,
                  train_dir=self.train_dir,
                  in_dim=4,  # hardcoded for image data
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
        self.trained_model.eval()

        dataloader_test = DataLoader(dataset_test, shuffle=False)

        z_list = []
        z_hat_list = []
        for batch_data in dataloader_test:
            x = batch_data[0]
            x_perm = batch_data[1]
            y = batch_data[2]
            y_perm = batch_data[3]
            z_gt = batch_data[4]

            x_torch = torch.cat((x, x_perm))
            y_torch = torch.squeeze(torch.cat((y, y_perm)))

            x_torch = x_torch.to(self.device)
            y_torch = y_torch.to(self.device)

            logits, h = self.trained_model(x_torch)

            h, h_perm = torch.split(h, split_size_or_sections=int(h.size()[0] / 2), dim=0)

            z_hat = h[:, 0, :].detach().cpu().numpy()

            z_list.append(z_gt.numpy())
            z_hat_list.append(z_hat)

        z = np.concatenate(z_list)
        z_hat = np.concatenate(z_hat_list)

        return z, z_hat
