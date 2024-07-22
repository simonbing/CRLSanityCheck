import random
import sys
import time

from absl import flags, app
import numpy as np
import torch
import wandb

from crc.baselines import TrainCMVAE

FLAGS = flags.FLAGS
flags.DEFINE_enum('model', None, ['cmvae'], 'Model to train.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1'], 'Dataset for training.')
flags.DEFINE_enum('experiment', None, ['scm_1'], 'Experiment for training.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')
# Shared hyperparameters
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')


class TrainApplication(object):
    def __init__(self, model, dataset, experiment, run_name, seed, batch_size,
                 epochs, lat_dim):
        self.model = model
        self.dataset = dataset
        self.experiment = experiment
        self.run_name = run_name
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.lat_dim = lat_dim
        trainer = self._get_trainer()
        self.trainer = trainer(dataset=self.dataset, experiment=self.experiment,
                               model=self.model, run_name=self.run_name,
                               seed=self.seed, batch_size=self.batch_size,
                               epochs=self.epochs, lat_dim=self.lat_dim)

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.trainer.train()

    def _get_trainer(self):
        if self.model == 'cmvae':
            return TrainCMVAE


def main(argv):
    if FLAGS.run_name is None:
        run_name = str(int(time.time()))
    else:
        run_name = FLAGS.run_name

    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        experiment=FLAGS.experiment,
        run_name=run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    gettrace = getattr(sys, 'gettrace', None)

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if not gettrace() else 'offline',  # don't log if debugging
        config=wandb_config
    )

    application = TrainApplication(model=FLAGS.model,
                                   dataset=FLAGS.dataset,
                                   experiment=FLAGS.experiment,
                                   run_name=run_name,
                                   seed=FLAGS.seed,
                                   batch_size=FLAGS.batch_size,
                                   epochs=FLAGS.epochs,
                                   lat_dim=FLAGS.lat_dim)
    application.run()


if __name__ == '__main__':
    app.run(main)
