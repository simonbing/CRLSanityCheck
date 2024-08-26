# import json
# import os
# import pickle
# import random
# import sys

from absl import flags, app
# import numpy as np
# import torch
#
# from crc.baselines import EvalCMVAE, EvalContrastCRL
# from crc.baselines.contrastive_crl.src.evaluation import compute_mccs, evaluate_graph_metrics
# from crc.eval import compute_MCC, mean_corr_coef_np, compute_SHD
# from crc.utils import NpEncoder

from crc.apps import EvalApplication

FLAGS = flags.FLAGS
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory for experiments.')
flags.DEFINE_enum('model', None, ['cmvae', 'contrast_crl'], 'Model to evaluate.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'contrast_synth'], 'Dataset for training.')
flags.DEFINE_string('experiment', None, 'Experiment for training.')
flags.DEFINE_string('run_name', None, 'Run name where trained model is saved.')
flags.DEFINE_list('metrics', None, 'Evaluation metrics to calculate.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


# class EvalApplication(object):
#     def __init__(self, seed, model, root_dir, dataset, experiment, run_name, metrics):
#         self.seed = seed
#         self.model = model
#         self.dataset = dataset
#         self.model_dir = os.path.join(root_dir, dataset, experiment, self.model)
#         self.train_dir = os.path.join(self.model_dir, run_name,
#                                       f'seed_{self.seed}', 'train')
#         self.eval_dir = os.path.join(self.model_dir, run_name,
#                                      f'seed_{self.seed}', 'eval')
#         if not os.path.exists(self.eval_dir):
#             os.makedirs(self.eval_dir)
#
#         evaluator = self._get_evaluator()
#         trained_model_path = os.path.join(self.train_dir, 'best_model.pt')
#         self.evaluator = evaluator(trained_model_path=trained_model_path)
#         self.metrics = metrics
#
#     def run(self):
#         # Set all seeds
#         torch.manual_seed(self.seed)
#         np.random.seed(self.seed)
#         random.seed(self.seed)
#
#         # Load test data (all ground truth info should be here!)
#         if self.dataset.endswith('synth'):
#             dataset_test_path = os.path.join(self.model_dir, f'test_dataset_seed_{self.seed}.pkl')
#         else:
#             dataset_test_path = os.path.join(self.model_dir, 'test_dataset.pkl')
#         with open(dataset_test_path, 'rb') as f:
#             dataset_test = pickle.load(f)
#
#         # Evaluate all metrics
#         results = {}
#         if 'MCC' in self.metrics:
#             z, z_hat = self.evaluator.get_encodings(dataset_test)
#             # TODO decide which one to keep
#             # mcc1 = compute_MCC(z_hat, z)
#             z_pred_sign_matched = z_hat * np.sign(z_hat)[:, 0:1] * np.sign(z)[:, 0:1]
#             # mccs = compute_mccs(z, z_hat)
#             # mccs_sign_matched = compute_mccs(z, z_pred_sign_matched)
#             # mccs_abs = compute_mccs(np.abs(z), np.abs(z_hat))
#             mcc, _, permutation = mean_corr_coef_np(z, z_hat)
#
#             results['mcc'] = float(mcc)
#
#             print(f'MCC: {mcc}')
#         if 'SHD' in self.metrics:
#             G, G_hat = self.evaluator.get_adjacency_matrices(dataset_test)
#
#             nr_edges = np.count_nonzero(G)
#             shd_dict = evaluate_graph_metrics(G, G_hat, nr_edges=nr_edges)
#
#             print(f"SHD: {shd_dict['SHD']}")
#             print(f"SHD_opt: {shd_dict['SHD_opt']}")
#             print(f"SHD_edge_matched: {shd_dict['SHD_edge_matched']}")
#
#             try: # Compute SHD with permutation from MCC
#                 G_hat_perm = G_hat[permutation[1], :][:, permutation[1]]
#                 shd_dict_perm = evaluate_graph_metrics(G, G_hat_perm, nr_edges=nr_edges)
#
#                 print(f"SHD perm: {shd_dict_perm['SHD']}")
#                 print(f"SHD_opt_perm: {shd_dict_perm['SHD_opt']}")
#                 print(f"SHD_edge_matched_perm: {shd_dict_perm['SHD_edge_matched']}")
#
#                 shd_dict['SHD_perm'] = shd_dict_perm['SHD']
#                 shd_dict['SHD_opt_perm'] = shd_dict_perm['SHD_opt']
#                 shd_dict['SHD_edge_matched_perm'] = shd_dict_perm['SHD_edge_matched']
#             except UnboundLocalError:
#                 pass
#
#             results = results | shd_dict
#
#         # Save results
#         with open(os.path.join(self.eval_dir, 'results.json'), 'w') as f:
#             json.dump(results, f, indent=4, cls=NpEncoder)
#
#     def _get_evaluator(self):
#         if self.model == 'cmvae':
#             return EvalCMVAE
#         elif self.model == 'contrast_crl':
#             return EvalContrastCRL


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
