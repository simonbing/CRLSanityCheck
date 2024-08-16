import json
import os
import pickle
import random

from absl import flags, app
import numpy as np
import torch

from crc.baselines import EvalCMVAE, EvalContrastCRL
from crc.eval import compute_MCC, mean_corr_coef_np, compute_SHD

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory for experiments.')
flags.DEFINE_enum('model', None, ['cmvae', 'contrast_crl'], 'Model to evaluate.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'contrast_synth'], 'Dataset for training.')
flags.DEFINE_enum('experiment', None, ['scm_1'], 'Experiment for training.')
flags.DEFINE_string('run_name', None, 'Run name where trained model is saved.')
flags.DEFINE_list('metrics', None, 'Evaluation metrics to calculate.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


class EvalApplication(object):
    def __init__(self, seed, model, root_dir, dataset, experiment, run_name, metrics):
        self.seed = seed
        self.model = model
        self.model_dir = os.path.join(root_dir, dataset, experiment, self.model)
        self.train_dir = os.path.join(self.model_dir, run_name, 'train')
        self.eval_dir = os.path.join(self.model_dir, run_name, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        evaluator = self._get_evaluator()
        trained_model_path = os.path.join(self.train_dir, 'best_model.pt')
        self.evaluator = evaluator(trained_model_path=trained_model_path)
        self.metrics = metrics

    def run(self):
        # Set all seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Load test data (all ground truth info should be here!)
        dataset_test_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        with open(dataset_test_path, 'rb') as f:
            dataset_test = pickle.load(f)

        # Evaluate all metrics
        results = {}
        if 'SHD' in self.metrics:
            G, G_hat = self.evaluator
            shd, shd_opt = compute_SHD(G, G_hat)

            results['shd'] = shd
            results['shd_opt'] = shd_opt

            print(f'SHD: {shd}')
            print(f'SHD_opt: {shd_opt}')
        elif 'MCC' in self.metrics:
            z, z_hat = self.evaluator.get_encodings(dataset_test)
            # TODO decide which one to keep
            # mcc1 = compute_MCC(z_hat, z)
            mcc, _, _ = mean_corr_coef_np(z, z_hat)

            results['mcc'] = mcc

            print(f'MCC: {mcc}')

        # Save results
        with open(os.path.join(self.eval_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def _get_evaluator(self):
        if self.model == 'cmvae':
            return EvalCMVAE
        elif self.model == 'contrast_crl':
            return EvalContrastCRL


def main(argv):
    application = EvalApplication(seed=FLAGS.seed,
                                  model=FLAGS.model,
                                  root_dir=FLAGS.root_dir,
                                  dataset=FLAGS.dataset,
                                  experiment=FLAGS.experiment,
                                  run_name=FLAGS.run_name,
                                  metrics=FLAGS.metrics)
    application.run()


if __name__ == '__main__':
    app.run(main)
