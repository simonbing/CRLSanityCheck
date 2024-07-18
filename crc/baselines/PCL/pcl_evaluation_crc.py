"""
Adapted from original PCL evaluation script
"""
import os
import pickle

import numpy as np
from absl import flags, app
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torch

from causalchamber.datasets import Dataset as ChamberDataset
from pcl import pcl, utils
from subfunc.showdata import *

FLAGS = flags.FLAGS

# Data
flags.DEFINE_string('data_root', None, 'Root directory where data is saved.')
flags.DEFINE_string('dataset', None, 'Dataset name.')
flags.DEFINE_string('experiment', None, 'Experiment name.')
flags.DEFINE_list('gt_features', None, 'Names of ground truth sources in data.')
flags.DEFINE_list('mix_features', None, 'Names of mixture features in data.')
# Eval
flags.DEFINE_string('train_dir', './storage', 'Training directory.')


class MainApplication(object):
    def __init__(self):
        # Data
        self.data_root = FLAGS.data_root
        self.dataset = FLAGS.dataset
        self.experiment = FLAGS.experiment
        self.gt_features = FLAGS.gt_features
        self.mix_features = FLAGS.mix_features
        # Eval
        self.train_dir = FLAGS.train_dir
        self.eval_dir = os.path.join(self.train_dir, 'model')

    def _get_data(self):
        dataset = ChamberDataset(self.dataset, root=self.data_root, download=True)

        experiment = dataset.get_experiment(self.experiment)
        df = experiment.as_pandas_dataframe()

        # Select features
        x_gt = df[self.gt_features].to_numpy()
        x_mix = df[self.mix_features].to_numpy()

        return x_gt, x_mix

    def _get_model(self):
        parmpath = os.path.join(self.eval_dir, 'parm.pkl')

        if self.eval_dir.find('.tar.gz') >= 0:
            unzipfolder = os.path.join(self.train_dir, 'temp_unzip')
            utils.unzip(self.eval_dir, unzipfolder)
            self.eval_dir = unzipfolder
            parmpath = os.path.join(unzipfolder, 'parm.pkl')

        modelpath = os.path.join(self.eval_dir, 'model.pt')

        # Load parameter file
        with open(parmpath, 'rb') as f:
            model_parm = pickle.load(f)

        list_hidden_nodes = model_parm['list_hidden_nodes']
        in_dim = model_parm['in_dim']
        latent_dim = model_parm['num_comp']
        self.ar_order = model_parm['ar_order']

        model = pcl.Net(h_sizes=list_hidden_nodes,
                        in_dim=in_dim,
                        latent_dim=latent_dim,
                        ar_order=self.ar_order)

        # TODO: might make more sense to move this to GPU
        self.device = 'cpu'
        model = model.to(self.device)
        model.train()

        # load parameters
        print('Load trainable parameters from %s...' % modelpath)
        checkpoint = torch.load(modelpath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['ema_state_dict'])

        return model

    def run(self):
        # Get data
        x_gt, x_mix = self._get_data()

        # Preprocessing
        pca = PCA(whiten=True)
        x_mix = pca.fit_transform(x_mix)

        # Evaluate model
        model = self._get_model()

        # Prepare data
        t_idx = np.arange(x_mix.shape[0] - self.ar_order) + self.ar_order
        t_idx = t_idx.reshape([-1, 1]) + np.arange(0, -self.ar_order - 1, -1).reshape([1, -1])
        xt_mix = x_mix[t_idx.reshape(-1), :].reshape([-1, self.ar_order + 1, x_mix.shape[-1]])
        xast_mix = xt_mix.copy()
        for i in range(self.ar_order):
            xast_mix[:, i + 1, :] = x_mix[np.random.choice(x_mix.shape[0], xt_mix.shape[0]), :]

        x_mix_torch = torch.from_numpy(np.concatenate([xt_mix, xast_mix], axis=0).astype(np.float32)).to(self.device)
        y_torch = torch.cat([torch.ones([xt_mix.shape[0]]), torch.zeros([xast_mix.shape[0]])]).to(self.device)

        logits, h = model(x_mix_torch)
        pred = (logits > 0).float()
        h, hast = torch.split(h, split_size_or_sections=int(h.size()[0] / 2), dim=0)

        # convert to numpy
        pred_val = pred.cpu().numpy()
        y_val = y_torch.cpu().numpy()
        h_val = np.squeeze(h[:, 0, :].detach().cpu().numpy())

        # correlation
        corrmat, sort_idx, _ = utils.correlation(h_val, x_gt[1:, :], 'Pearson')
        h_val_sort = h_val[:, sort_idx] * np.sign(np.diag(corrmat))[None, :]

        meanabscorr = np.mean(np.abs(np.diag(corrmat)))

        # TODO: plot correlation matrix

        # Print results
        print('Result...')
        print('    accuracy (train) : %7.4f [percent]' % (accuracy_score(pred_val, y_val.T) * 100))
        print('    correlation      : %7.4f' % meanabscorr)

        # Visualization
        # Original source
        showtimedata(x_gt, figsize=[1, 1], linewidth=2.5)

        # Estimation
        showtimedata(h_val_sort, figsize=[1, 1], linewidth=2.5)
        a=0


def main(argv):
    application = MainApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)