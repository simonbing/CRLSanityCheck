import os
import pickle

import torch
from torch.utils.data import DataLoader

from crc.baselines.contrastive_crl.src.data_generation import get_data_from_kwargs
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

        # TODO: sample test data (5000 samples)
        # Get data
        dataset_train, dataset_val, dataset_test = get_chamber_data(dataset=self.dataset, seed=self.seed)

        # Make dataloaders
        dl_train = DataLoader(dataset_train, shuffle=True, batch_size=self.batch_size)
        dl_val = DataLoader(dataset_val, shuffle=False, batch_size=self.batch_size)

        # Save train data (as torch dataset)
        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path):
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)

        val_data_path = os.path.join(self.model_dir, 'val_dataset.pkl')
        if not os.path.exists(val_data_path):
            with open(val_data_path, 'wb') as f:
                pickle.dump(dataset_val, f, protocol=pickle.HIGHEST_PROTOCOL)

        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path):
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataset_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save training config metadata
        # TODO



        # Build model
        model = self._get_model()

        training_kwargs = {
            'epochs': self.epochs,
            'optimizer_name': 'adam',
            'mu': 10e-5,
            'eta': 10e-4,
            'kappa': 1.0, # TODO: figure out the correct value for this
            'lr_nonparametric': 5*10e-4
        }

        # Train model
        best_model, last_model, _, _ = train_model(model, device, dl_train, dl_val,
                                 training_kwargs)
        # Save model
        torch.save(best_model, os.path.join(self.train_dir, 'best_model.pt'))
        torch.save(last_model, os.path.join(self.train_dir, 'last_model.pt'))


class EvalContrastCRL(EvalModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_adjacency_matrices(self, dataset_test):
        G = dataset_test.W
        G_hat = self.trained_model.parametric_part.A.t().cpu().detach().numpy()

        return G, G_hat

    def get_encodings(self, dataset_test):
        self.trained_model.eval()

        z_gt = dataset_test.z_obs
        x_gt = dataset_test.f(torch.tensor(z_gt, dtype=torch.float)).to(self.device)

        z_hat = self.trained_model.get_z(x_gt).cpu().detach().numpy()

        return z_gt, z_hat
