import os
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch

from crc.wrappers import TrainModel, EvalModel
from crc.baselines.rgb_baseline.src import ChamberDataset, RGBBaseline


class TrainRGBBaseline(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Check if trained model already exists, skip training if so
        if os.path.exists(os.path.join(self.train_dir, 'best_model.pt')):
            print('Trained model found, skipping training!')
            return

        # Get data
        dataset = ChamberDataset(dataset=self.dataset, task=self.task,
                                 data_root=self.data_root)
        train_idxs, test_idxs = train_test_split(range(len(dataset)),
                                                 train_size=0.8,
                                                 shuffle=True,
                                                 random_state=self.seed)
        dataset_train = Subset(dataset, train_idxs)
        dataset_test = Subset(dataset, test_idxs)

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

        # Save model
        model = RGBBaseline()
        torch.save(model, os.path.join(self.train_dir, 'best_model.pt'))


class EvalRGBBaseline(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        self.trained_model.eval()

        dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)

        z_list = []
        z_hat_list = []
        # Iterate over dataset, get encoding
        for batch in dataloader_test:
            x_batch = batch[0]
            z_batch = batch[1]

            x_batch = x_batch.to(self.device)

            z_hat_batch = self.trained_model.get_z(x_batch)

            z_list.append(z_batch)
            z_hat_list.append(z_hat_batch)

        z_gt = torch.cat(z_list).cpu().detach().numpy()
        z_hat = torch.cat(z_hat_list).cpu().detach().numpy()

        return z_gt, z_hat
