from crc.ood_estimation.base_estimator import OODEstimator


class MLPOODEstimator(OODEstimator):
    def __init__(self, seed, task):
        super().__init__(seed, task)

    def train(self):
        pass

    def predict(self, X_ood):
        pass
