from crc.ood_estimation.base_estimator import OODEstimator


class CRLOODEstimator(OODEstimator):
    def __init__(self, seed, task, data_root):
        super().__init__(seed, task, data_root)

    def train(self):
        pass

    def predict(self, X_ood):
        pass
