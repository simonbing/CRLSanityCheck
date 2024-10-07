from crc.ood_estimation.base_estimator import OODEstimator
from crc.baselines import TrainCMVAE, TrainContrastCRL


class CRLOODEstimator(OODEstimator):
    def __init__(self, seed, task, data_root, crl_model):
        super().__init__(seed, task, data_root)

        self.crl_model = crl_model

        # Get CRL trainer
        trainer = self._get_trainer()
        self.trainer = trainer(data_root=self.data_root, dataset=None,
                               experiment=None, overwrite_data=None,
                               model=self.crl_model, run_name=None,
                               seed=self.seed, batch_size=None, epochs=None,
                               lat_dim=None, root_dir=None)

    def _get_trainer(self):
        if self.crl_model == 'cmvae':
            return TrainCMVAE
        elif self.crl_model == 'contrast_crl':
            return TrainContrastCRL

    def train(self):
        pass

    def predict(self, X_ood):
        pass
