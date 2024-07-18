"""
Adapted from original PCL training script
"""
import os
import pickle
import tarfile
import shutil

from absl import flags, app
from sklearn.decomposition import PCA

from causalchamber.datasets import Dataset as ChamberDataset
from pcl.train import train
from subfunc.showdata import showtimedata

FLAGS = flags.FLAGS
# Data
flags.DEFINE_string('data_root', None, 'Root directory where data is saved.')
flags.DEFINE_string('dataset', None, 'Dataset name.')
flags.DEFINE_string('experiment', None, 'Experiment name.')
flags.DEFINE_list('gt_features', None, 'Names of ground truth sources in data.')
flags.DEFINE_list('mix_features', None, 'Names of mixture features in data.')
# Training
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_multi_integer('mlp_layers', [128, 128], 'Hidden dims of MLP.')
flags.DEFINE_float('init_lr', 0.1, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for SGD.')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')
flags.DEFINE_float('decay_factor', 0.1, 'Decay factor.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('moving_average_decay', 0.999, 'Moving average decay rate.')
flags.DEFINE_string('train_dir', './storage', 'Training directory.')


class MainApplication(object):
    def __init__(self):
        # Data
        self.data_root = FLAGS.data_root
        self.dataset = FLAGS.dataset
        self.experiment = FLAGS.experiment
        self.gt_features = FLAGS.gt_features
        self.mix_features = FLAGS.mix_features
        # Training
        self.seed = FLAGS.seed
        self.mlp_layers = FLAGS.mlp_layers
        self.init_lr = FLAGS.init_lr
        self.momentum = FLAGS.momentum
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.decay_factor = FLAGS.decay_factor
        self.weight_decay = FLAGS.weight_decay
        self.train_dir = os.path.join(FLAGS.train_dir, 'model')
        self.moving_average_decay = FLAGS.moving_average_decay

    def _get_data(self):
        dataset = ChamberDataset(self.dataset, root=self.data_root, download=True)

        experiment = dataset.get_experiment(self.experiment)
        df = experiment.as_pandas_dataframe()

        # Select features
        x_gt = df[self.gt_features].to_numpy()
        x_mix = df[self.mix_features].to_numpy()

        # showtimedata(x_gt[:8000, :], figsize=[1, 1], linewidth=2.5)
        # showtimedata(x_mix[:8000, :], figsize=[1, 1], linewidth=2.5)

        return x_gt, x_mix

    def run(self):
        # Get data
        x_gt, x_mix = self._get_data()

        # Preprocessing
        pca = PCA(whiten=True)
        x_mix = pca.fit_transform(x_mix)

        # Training preparation
        list_hidden_nodes = self.mlp_layers + [x_gt.shape[1]]
        max_steps = int(x_mix.shape[0] / self.batch_size * self.epochs)

        # Prepare output directory
        if self.train_dir.find('/storage/') > -1:
            if os.path.exists(self.train_dir):
                print('delete savefolder: %s...' % self.train_dir)
                shutil.rmtree(self.train_dir)  # remove folder
            print('make savefolder: %s...' % self.train_dir)
            os.makedirs(self.train_dir)  # make folder
        else:
            assert False, 'savefolder looks wrong'

        # Train model
        train(x_mix,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=self.init_lr,
              momentum=self.momentum,
              max_steps=max_steps,
              decay_steps=int(max_steps / 2),
              decay_factor=self.decay_factor,
              batch_size=self.batch_size,
              train_dir=self.train_dir,
              latent_dim=x_gt.shape[1],
              ar_order=1,
              weight_decay=self.weight_decay,
              checkpoint_steps=2*max_steps,
              moving_average_decay=self.moving_average_decay,
              summary_steps=int(max_steps / 10),
              random_seed=self.seed
        )

        # Save model
        model_parm = {'random_seed': self.seed,
                      'num_comp': x_gt.shape[1],
                      'num_data': x_gt.shape[0],
                      'in_dim': x_mix.shape[1],
                      'ar_order': 1,
                      'model': 'pca',
                      'list_hidden_nodes': list_hidden_nodes,
                      'moving_average_decay': self.moving_average_decay}

        print('Save parameters...')
        saveparmpath = os.path.join(self.train_dir, 'parm.pkl')
        with open(saveparmpath, 'wb') as f:
            pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

        # Save as tarfile
        tarname = self.train_dir + ".tar.gz"
        archive = tarfile.open(tarname, mode="w:gz")
        archive.add(self.train_dir, arcname="./")
        archive.close()

        print('done.')


def main(argv):
    application = MainApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)
