import logging
import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
import wandb

from crc.ood_estimation.base_estimator import OODEstimator
from crc.baselines import TrainPCL, TrainCMVAE, TrainContrastCRL
from crc.baselines import EvalPCL, EvalCMVAE, EvalContrastCRL


class CRLOODEstimator(OODEstimator):
    def __init__(self, seed, task, dataset, data_root, results_root, crl_model, lat_dim,
                 batch_size,
                 epochs, run_name,
                 overwrite_data=False):
        super().__init__(seed, task, data_root, results_root)
        self.dataset = dataset
        self.lat_dim = lat_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.crl_model = crl_model

        self.run_name = run_name
        self.overwrite_data = overwrite_data

        # Get CRL trainer
        trainer = self._get_trainer()
        self.trainer = trainer(data_root=self.data_root, dataset=self.dataset,
                               task=self.task, overwrite_data=self.overwrite_data,
                               model=self.crl_model, run_name=self.run_name,
                               seed=self.seed, batch_size=self.batch_size,
                               epochs=self.epochs,
                               lat_dim=self.lat_dim, root_dir=self.results_root)

        # Linear head
        self.lin_model = LinearRegression()

    def _get_trainer(self):
        if self.crl_model == 'cmvae':
            return TrainCMVAE
        elif self.crl_model == 'contrast_crl':
            return TrainContrastCRL
        elif self.crl_model == 'pcl':
            return TrainPCL

    def _get_evaluator(self):
        if self.crl_model == 'cmvae':
            return EvalCMVAE
        elif self.crl_model == 'contrast_crl':
            return EvalContrastCRL
        elif self.crl_model == 'pcl':
            return EvalPCL

    def train(self, X, y):
        self.trainer.train()

        # Get CRL evaluator
        evaluator = self._get_evaluator()
        self.evaluator = evaluator(trained_model_path=os.path.join(self.trainer.train_dir,
                                                                   'best_model.pt'))

        dataset_train_path = os.path.join(self.trainer.model_dir, 'train_dataset.pkl')
        with open(dataset_train_path, 'rb') as f:
            dataset_train = pickle.load(f)

        _, z_hat_train = self.evaluator.get_encodings(dataset_train)
        # Get embeddings of training data the same way as in eval (maybe just reuse that function)
        # but then how do we make sure it is not shuffled and fucks up the labels?
        # -> dont shuffle in the dataloader!
        num_train_samples = int(len(y) * self.train_frac)
        y_train = y[:num_train_samples]
        y_test = y[num_train_samples:]
        assert len(y_train) == len(z_hat_train)

        # Use embeddings to train linear head

        # Train linear regression with embedding and labels
        self.lin_model.fit(z_hat_train[1:, :], y_train[1:, :])

        # Get test dataset
        dataset_test_path = os.path.join(self.trainer.model_dir, 'test_dataset.pkl')
        with open(dataset_test_path, 'rb') as f:
            dataset_test = pickle.load(f)
        _, z_hat_test = self.evaluator.get_encodings(dataset_test)

        y_hat_test = self.lin_model.predict(z_hat_test)
        mse_id = np.mean((y_hat_test - y_test) ** 2)
        logging.info(f'ID mse: {mse_id}')
        wandb.run.summary['mse_id'] = mse_id

    def predict(self, X_ood):
        # Load trained model

        # Convert df into something the model can ingest (image loading) and ret
        # Probably need to make a new dataset somewhere earlier (where we get the task)
        # and save this for loading later... Maybe in the init of this app?
        # Should be able to pass this task to the respective dataset building methods of each model...

        # Then: embed from that dataset, predict using trained linear head

        y_hat = self.lin_model.predict(Z_ood)
        pass
