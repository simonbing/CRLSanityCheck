import logging

import numpy as np
from sklearn.model_selection import train_test_split
import wandb

from crc.ood_estimation.base_estimator import OODEstimator


class MeanEstimator(OODEstimator):
    def __init__(self, seed, task, data_root):
        super().__init__(seed, task, data_root)

    def train(self, X, y):
        y_train, y_val = train_test_split(y,
                                          train_size=self.train_frac,
                                          shuffle=True,
                                          random_state=self.seed)

        self.model = np.mean(y_train)

        mse_val = np.mean((y_val - self.model) ** 2)
        logging.info(f'ID mse: {mse_val}')
        wandb.run.summary['mse_id'] = mse_val

    def predict(self, X_ood):
        return [self.model] * len(X_ood)
