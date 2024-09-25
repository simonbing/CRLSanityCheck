import random

import numpy as np
import torch

from crc.ood_estimation import OLSOODEstimator


class OODEstimatorApplication(object):
    def __init__(self, seed, estimation_model, task):
        self.seed = seed
        self.estimator = self._get_estimator(estimation_model)
        self.task = task  # encodes which environments to train and test on

    def _get_estimator(self, estimation_model):
        if estimation_model == 'ols':
            return OLSOODEstimator(seed=self.seed)

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Train estimator on training environments
        self.estimator.train()

        # Get estimates on held out test environment
        y_hat = self.estimator.predict()

        # Compute held out test metrics
