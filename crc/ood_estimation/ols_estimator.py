import logging

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import wandb

from crc.ood_estimation.base_estimator import OODEstimator


class OLSOODEstimator(OODEstimator):
    def __init__(self, seed, task, data_root):
        image_data = False
        super().__init__(seed, task, data_root)

        self.model = LinearRegression()

    def train(self, X, y):
        # Discard image info, convert directly to np array
        X = X.drop(columns='image_file').to_numpy()
        ###
        # experimental linear mixing of g.t. signal
        # W = np.array([[0.3, 0.1, 0.2, 0.1, 0.2],
        #               [0.5, 0.1, 0.1, 0.2, 0.1],
        #               [0.4, 0.1, 0.2, 0.1, 0.2],
        #               [0.1, 0.1, 0.1, 0.1, 0.6],
        #               [0.2, 0.1, 0.3, 0.3, 0.1]])
        # X = X @ W
        ###
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          train_size=self.train_frac,
                                                          shuffle=True,
                                                          random_state=self.seed)

        self.model.fit(X_train, y_train)

        y_hat = self.model.predict(X_val)
        mse_val = np.mean((y_val - y_hat) ** 2)
        logging.info(f'ID mse: {mse_val}')
        wandb.run.summary['mse_id'] = mse_val

    def predict(self, X_ood):
        # Discard image info, convert directly to np array
        X_ood = X_ood.drop(columns='image_file').to_numpy()
        ###
        # experimental linear mixing of gt signal
        # W = np.array([[0.3, 0.1, 0.2, 0.1, 0.2],
        #               [0.5, 0.1, 0.1, 0.2, 0.1],
        #               [0.4, 0.1, 0.2, 0.1, 0.2],
        #               [0.1, 0.1, 0.1, 0.1, 0.6],
        #               [0.2, 0.1, 0.3, 0.3, 0.1]])
        # X_ood = X_ood @ W
        ###

        y_hat = self.model.predict(X_ood)

        return y_hat


class LassoOODEstimator(OLSOODEstimator):
    def __init__(self, seed, task, data_root):
        super().__init__(seed, task, data_root)

        self.model = Lasso(alpha=0.1)  # alpha value hardcoded for now
