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
        if self.dataset == 'contrastive_synth':
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
        # Get data
        dataloader_obs, dataloader_int, \
            dataloader_obs_val, dataloader_int_val = \
            get_chamber_data(dataset=self.dataset,
                             seed=self.seed,
                             batch_size=self.batch_size)

        # # Prepare data for training
        # mixing = 'mlp' # TODO: this can also be 'image', make this an argument
        # data_kwargs = {
        #     'mixing': mixing,
        #     'd': None,
        #     'k': None,
        #     'n': None,
        #     'seed': self.seed,
        #     'dim_x': None,
        #     'hidden_dim': None,
        #     'hidden_layers': None
        # } # TODO get these kwargs
        # databag = get_data_from_kwargs(data_kwargs) # databags is the term used in original code

        # Save training config metadata
        # TODO

        # Save train data (as torch dataset)
        # TODO

        # Build model
        model = self._get_model()

        # Train model
        _, _, _, _ = train_model()
