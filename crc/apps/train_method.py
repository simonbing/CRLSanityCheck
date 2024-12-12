import os
import pickle
import random

from absl import app, flags
import numpy as np
import torch
from torch.utils.data import Subset
import wandb

from crc.utils import train_val_test_split
from crc.methods.utils import get_method


class TrainMethod(object):
    def __init__(self, method, out_dir, run_name, overwrite=False, **kwargs):
        self.method_name = method

        self.method = get_method(method=method)(**kwargs)

        self.model_dir = os.path.join(out_dir, kwargs['dataset'], kwargs['task'],
                                      self.method_name)
        self.train_dir = os.path.join(self.model_dir, run_name,
                                      f"seed_{kwargs['seed']}", 'train')
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.overwrite = overwrite

    def run(self):
        # Check if trained model already exists, skip training if so
        if os.path.exists(os.path.join(self.train_dir, 'best_model.pt')):
            print('Trained model found, skipping training!')
            return

        # Get datasets and save
        dataset = self.method.get_dataset()

        # Split into train/val/test
        train_idxs, val_idxs, test_idxs = train_val_test_split(
            np.arange(len(dataset)),
            train_size=0.8,  # hardcoded
            val_size=0.1,
            random_state=42)  # this is fixed across all wrongs, to ensure that
                              # the same train/val/test split is used for all models

        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, val_idxs)
        test_dataset = Subset(dataset, test_idxs)

        train_data_path = os.path.join(self.model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path) or self.overwrite:
            with open(train_data_path, 'wb') as f:
                pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        val_data_path = os.path.join(self.model_dir, 'val_dataset.pkl')
        if not os.path.exists(val_data_path) or self.overwrite:
            with open(val_data_path, 'wb') as f:
                pickle.dump(val_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path) or self.overwrite:
            with open(test_data_path, 'wb') as f:
                pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Training
        best_model = self.method.train(train_dataset, val_dataset)
        print('Training finished!')

        # Save model
        torch.save(best_model, os.path.join(self.train_dir, 'best_model.pt'))


def main(argv):
    FLAGS = flags.FLAGS

    flags.DEFINE_integer('seed', 0, 'Random seed.')
    flags.DEFINE_string('out_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                        'Directory to save results to.')
    flags.DEFINE_enum('method', 'multiview_iv', ['multiview_iv', 'contrast_crl'],
                      'Representation learning method')
    flags.DEFINE_enum('dataset', 'lt_camera_v1', ['lt_camera_v1'], 'Dataset.')
    flags.DEFINE_enum('task', 'lt_scm_2', ['lt_scm_2'], 'Experimental task.')
    flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/data/chamber_downloads',
                        'Root directory of data.')
    flags.DEFINE_string('run_name', None, 'Name of experimental run.')
    flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')
    flags.DEFINE_enum('encoder', 'conv', ['fc', 'conv'], 'Encoder type.')
    flags.DEFINE_integer('bs', 512, 'Batch size.')
    flags.DEFINE_integer('epochs', 10, 'Training epochs.')
    flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
    # Multiview flags
    flags.DEFINE_integer('n_envs', 1, 'Number of interventional environments for multiview data.')
    flags.DEFINE_enum('selection', 'ground_truth', ['ground_truth'], 'Selection for estimating content indices.')
    flags.DEFINE_float('tau', 1.0, 'Temperature parameter for multiview loss.')

    # Multiview Args
    kwarg_dict = {'n_envs': FLAGS.n_envs,
                  'selection': FLAGS.selection,
                  'tau': FLAGS.tau}

    WANDB = False

    wandb_config = dict(
        model=FLAGS.method,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.bs,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if WANDB else 'offline',
        # don't log if debugging
        config=wandb_config
    )

    method = TrainMethod(method=FLAGS.method, out_dir=FLAGS.out_dir, seed=FLAGS.seed,
                         dataset=FLAGS.dataset, task=FLAGS.task,
                         data_root=FLAGS.data_set, run_name=FLAGS.run_name, d=FLAGS.lat_dim,
                         batch_size=FLAGS.bs, epochs=FLAGS.epochs,
                         lr=FLAGS.lr, encoder=FLAGS.encoder, **kwarg_dict)

    method.run()


if __name__ == '__main__':
    app.run(main)
