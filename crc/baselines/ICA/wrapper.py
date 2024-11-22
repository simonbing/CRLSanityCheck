import joblib
import os
import pickle


from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from torch.utils.data import Subset, DataLoader

from crc.wrappers import TrainModel, EvalModel
from crc.baselines.ICA.src import ChamberDataset
from crc.utils import  get_device


class TrainICA(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        # Check if trained model already exists, skip training if so
        if os.path.exists(os.path.join(self.train_dir, 'best_model.pkl')):
            print('Trained model found, skipping training!')
            return

        # Get dataset
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

        # Train data
        # Need entire dataset at once, hence batch size
        dataloader_train = DataLoader(dataset_train, batch_size=len(train_idxs), shuffle=True)

        X_train, _ = next(iter(dataloader_train))

        model = FastICA(n_components=self.lat_dim, random_state=self.seed)

        model.fit(X_train.numpy())

        # Save model
        joblib.dump(model, os.path.join(self.train_dir, 'best_model.pkl'))


class EvalICA(EvalModel):
    def __init__(self, trained_model_path):
        self.trained_model = joblib.load(trained_model_path)

    def get_adjacency_matrices(self, dataset_test):
        pass

    def get_encodings(self, dataset_test):
        dataloader_test = DataLoader(dataset_test, batch_size=100000, shuffle=False)

        X_test, z_gt = next(iter(dataloader_test))

        z_hat = self.trained_model.transform(X_test.numpy())

        return z_gt.numpy(), z_hat
