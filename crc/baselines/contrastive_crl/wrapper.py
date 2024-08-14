from crc.baselines.contrastive_crl.src.data_generation import get_data_from_kwargs
from crc.baselines.contrastive_crl.src.utils import get_chamber_data
from crc.baselines.contrastive_crl.src.models import build_model_from_kwargs
from crc.baselines.contrastive_crl.src.training import train_model

from crc.wrappers import TrainModel, EvalModel
from crc.utils import get_device


class TrainContrastCRL(TrainModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self):
        """
        Adapted from source code for "Learning Linear Causal Representations
        from Interventions under General Nonlinear Mixing".
        """
        # Get data
        get_chamber_data(dataset=self.dataset, seed=self.seed, batch_size=self.batch_size)

        # Prepare data for training
        mixing = 'mlp' # TODO: this can also be 'image', make this an argument
        data_kwargs = {
            'mixing': mixing,
            'd': None,
            'k': None,
            'n': None,
            'seed': self.seed,
            'dim_x': None,
            'hidden_dim': None,
            'hidden_layers': None
        } # TODO get these kwargs
        databag = get_data_from_kwargs(data_kwargs) # databags is the term used in original code

        # Save training config metadata
        # TODO

        # Save train data (as torch dataset)
        # TODO

        # Build model
        model_kwargs = {
            'type': 'contrastive',
            'input_dim': None,
            'latent_dim': self.lat_dim,
            'hidden_dim': None,
            'hidden_layers': None,
            'residual': True
        } # TODO get these

        image_data = False
        if image_data:
            model_kwargs['image'] = True

        model = build_model_from_kwargs(model_kwargs)

        # Train model
        _, _, _, _ = train_model()
