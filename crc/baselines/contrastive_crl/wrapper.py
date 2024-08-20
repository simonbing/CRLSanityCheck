import json
import os
import pickle

import torch
from torch.utils.data import DataLoader

from crc.baselines.contrastive_crl.src.data_generation import ContrastiveCRLDataset
from crc.baselines.contrastive_crl.src.utils import get_chamber_data
from crc.baselines.contrastive_crl.src.models import get_contrastive_synthetic, get_contrastive_image
from crc.baselines.contrastive_crl.src.training import train_model

from crc.wrappers import TrainModel, EvalModel
from crc.utils import get_device


class TrainContrastCRL(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_model(self):
        if self.dataset == 'contrast_synth':
            return get_contrastive_synthetic(input_dim=20, latent_dim=self.lat_dim,
                                             hidden_dim=512, hidden_layers=0,
                                             residual=True)
        else:
            return get_contrastive_image(latent_dim=self.lat_dim, channels=10)

    def train(self):
        """
        Adapted from source code for "Learning Linear Causal Representations
        from Interventions under General Nonlinear Mixing".
        """
        device = get_device()
        print(f'using device: {device}')

        # Get data
        dataset_train, dataset_val, dataset_test = get_chamber_data(dataset=self.dataset,
                                                                    exp=self.experiment,
                                                                    data_root=self.data_root,
                                                                    seed=self.seed)

        # Make dataloaders
        dl_train = DataLoader(dataset_train, shuffle=True, batch_size=self.batch_size)
        dl_val = DataLoader(dataset_val, shuffle=False, batch_size=self.batch_size)

        # Save train data (as torch dataset)
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path):
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save val data
        val_data_path = os.path.join(self.model_dir, 'val_dataset.pkl')
        if not os.path.exists(val_data_path):
            with open(val_data_path, 'wb') as f:
                pickle.dump(dataset_val, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save test data
        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path):
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataset_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Build model
        model = self._get_model()

        training_metadata = {
            'batch_size': self.batch_size,
            'lat_dim': self.lat_dim,
            'seed': self.seed
        }

        training_kwargs = {
            'epochs': self.epochs,
            'optimizer_name': 'adam',
            'mu': 10e-5,
            'eta': 10e-4,
            'kappa': 0.1,
            'lr_nonparametric': 5*10e-4,
            'weight_decay': 0.0
        }

        # Save training config metadata
        with open(os.path.join(self.train_dir, 'config.json'), 'w') as f:
            json.dump(training_metadata | training_kwargs, f, indent=4)

        # Train model
        best_model, last_model, _, _ = train_model(model, device, dl_train,
                                                   dl_val, training_kwargs,
                                                   verbose=True)
        # Save model
        torch.save(best_model, os.path.join(self.train_dir, 'best_model.pt'))
        torch.save(last_model, os.path.join(self.train_dir, 'last_model.pt'))

        print('Training finished!')


class EvalContrastCRL(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        try:
            G = dataset_test.W
        except AttributeError:
            G = dataset_test.dataset.W
        G_hat = self.trained_model.parametric_part.A.t().cpu().detach().numpy()

        return G, G_hat

    def get_encodings(self, dataset_test):
        self.trained_model.eval()

        if isinstance(dataset_test, ContrastiveCRLDataset):
            z_gt = dataset_test.z_obs.cpu().detach().numpy()
            x_gt = dataset_test.f(torch.tensor(z_gt, dtype=torch.float)).to(self.device)

            z_hat = self.trained_model.get_z(x_gt).cpu().detach().numpy()
        else:
            dataloader_test = DataLoader(dataset_test, shuffle=False)

            z_list = []
            z_hat_list = []
            # Iterate over test dataloader and encode all samples and save gt data
            for X in dataloader_test:
                x_obs = X[0]
                z_obs = X[3]

                x_obs = x_obs.to(self.device)

                z_hat_batch = self.trained_model.get_z(x_obs)

                z_list.append(z_obs)
                z_hat_list.append(z_hat_batch)

            z_gt = torch.cat(z_list).cpu().detach().numpy()
            z_hat = torch.cat(z_hat_list).cpu().detach().numpy()

        return z_gt, z_hat
