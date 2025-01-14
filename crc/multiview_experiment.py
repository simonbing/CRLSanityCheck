import os
import random
import sys

from absl import flags, app
import numpy as np
import torch
import wandb

from crc.utils import get_device

FLAGS = flags.FLAGS

# General params
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum('model', 'contrast_crl', ['contrast_crl', 'contrast_crl_linear'],
                  'Model to train.')
flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/'
                                 'CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory where output is saved.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'lt_camera_walks_v1',
                                    'contrast_synth', 'contrast_img',
                                    'contrast_semi_synth_mlp', 'contrast_synth_re'],
                  'Dataset for training.')
flags.DEFINE_string('task', None, 'Experimental task for training.')
flags.DEFINE_bool('overwrite_data', False, 'Overwrite existing saved data.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')

# Training params
# flags.DEFINE_enum('optim', 'adam', ['adam', 'sgd'], 'Optimizer for training.')
flags.DEFINE_float('lr', 0.0005, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('train_steps', 100000, 'Training steps.')
# flags.DEFINE_integer('epochs', 100, 'Training epochs.')


def main(argv):
    # wandb stuff
    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    gettrace = getattr(sys, 'gettrace', None)

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if not gettrace() else 'offline',
        # don't log if debugging
        config=wandb_config
    )

    # Training preparation
    # Set all seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # TODO: make all dirs

    device = get_device()
    print(f'using device: {device}')

    # Training
    step = 1
    loss_values = []  # list to keep track of loss values
    while step <= FLAGS.train_steps:
        # training step
        data = next(train_iterator)  # contains images, texts, and labels
        loss_value, _ = train_step(data, encoders, loss_func, optimizer, params,
                                   args=args)
        loss_values.append(loss_value)

    print('Training finished!')

    # Evaluation


if __name__ == '__main__':
    app.run(main)
