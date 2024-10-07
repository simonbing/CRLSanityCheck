import random

import numpy as np
import torch

from crc.baselines import TrainCMVAE, TrainContrastCRL


class TrainApplication(object):
    def __init__(self, model, output_root, data_root, dataset, task,
                 run_name, overwrite_data, seed, batch_size, epochs, lat_dim):
        self.model = model
        self.output_root = output_root
        self.data_root = data_root
        self.dataset = dataset
        self.task = task
        self.run_name = run_name
        self.overwrite_data = overwrite_data
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.lat_dim = lat_dim
        trainer = self._get_trainer()
        self.trainer = trainer(data_root=self.data_root, dataset=self.dataset,
                               task=self.task, overwrite_data=self.overwrite_data,
                               model=self.model, run_name=self.run_name,
                               seed=self.seed, batch_size=self.batch_size,
                               epochs=self.epochs, lat_dim=self.lat_dim,
                               root_dir=self.output_root)

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.trainer.train()

    def _get_trainer(self):
        if self.model == 'cmvae':
            return TrainCMVAE
        elif self.model == 'contrast_crl':
            return TrainContrastCRL
