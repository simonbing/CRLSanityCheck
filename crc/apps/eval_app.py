import json
import os
import pickle
import random

import numpy as np
import torch
import wandb

from crc.baselines import EvalCMVAE, EvalContrastCRL, EvalPCL
from crc.baselines.contrastive_crl.src.evaluation import compute_mccs, evaluate_graph_metrics
from crc.baselines.PCL.pcl.utils import correlation
from crc.eval import compute_MCC, mean_corr_coef_np, compute_SHD
from crc.utils import NpEncoder


class EvalApplication(object):
    def __init__(self, seed, model, root_dir, dataset, task, run_name, metrics):
        self.seed = seed
        self.model = model
        self.dataset = dataset
        self.model_dir = os.path.join(root_dir, dataset, task, self.model)
        self.train_dir = os.path.join(self.model_dir, run_name,
                                      f'seed_{self.seed}', 'train')
        self.eval_dir = os.path.join(self.model_dir, run_name,
                                     f'seed_{self.seed}', 'eval')
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
        if self.dataset.endswith('synth'):
            dataset_test_path = os.path.join(self.model_dir, f'test_dataset_seed_{self.seed}.pkl')
        else:
            dataset_test_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        with open(dataset_test_path, 'rb') as f:
            dataset_test = pickle.load(f)

        # Evaluate all metrics
        results = {}
        if 'MCC' in self.metrics:
            z, z_hat = self.evaluator.get_encodings(dataset_test)
            # Experimental PCL correlation
            mcc_pcl_mat, _, _ = correlation(z_hat, z, 'Pearson')
            mcc_pcl = np.mean(np.abs(np.diag(mcc_pcl_mat)))
            # TODO decide which one to keep
            # mcc1 = compute_MCC(z_hat, z)
            z_pred_sign_matched = z_hat * np.sign(z_hat)[:, 0:1] * np.sign(z)[:, 0:1]
            mccs = compute_mccs(z, z_hat)
            mccs_sign_matched = compute_mccs(z, z_pred_sign_matched)
            mccs_abs = compute_mccs(np.abs(z), np.abs(z_hat))
            mcc, _, permutation = mean_corr_coef_np(z, z_hat)

            results['mcc'] = float(mcc)
            results['mcc_w_in'] = mccs['mcc_w_in']
            results['mcc_w_out'] = mccs['mcc_w_out']
            results['mcc_pcl'] = mcc_pcl

            print(f'MCC: {mcc}')

            print('Alternative MCC scores:')
            print(mccs)
            print(mccs_sign_matched)
            print(mccs_abs)
        if 'SHD' in self.metrics:
            G, G_hat = self.evaluator.get_adjacency_matrices(dataset_test)

            nr_edges = np.count_nonzero(G)
            shd_dict = evaluate_graph_metrics(G, G_hat, nr_edges=nr_edges)

            print(f"SHD: {shd_dict['SHD']}")
            print(f"SHD_opt: {shd_dict['SHD_opt']}")
            print(f"SHD_edge_matched: {shd_dict['SHD_edge_matched']}")

            try: # Compute SHD with permutation from MCC
                G_hat_perm = G_hat[permutation[1], :][:, permutation[1]]
                shd_dict_perm = evaluate_graph_metrics(G, G_hat_perm, nr_edges=nr_edges)

                print(f"SHD perm: {shd_dict_perm['SHD']}")
                print(f"SHD_opt_perm: {shd_dict_perm['SHD_opt']}")
                print(f"SHD_edge_matched_perm: {shd_dict_perm['SHD_edge_matched']}")

                shd_dict['SHD_perm'] = shd_dict_perm['SHD']
                shd_dict['SHD_opt_perm'] = shd_dict_perm['SHD_opt']
                shd_dict['SHD_edge_matched_perm'] = shd_dict_perm['SHD_edge_matched']
            except UnboundLocalError:
                pass

            results = results | shd_dict

        # Log to wandb summary
        for key in results:
            wandb.run.summary[key] = results[key]

        # Save results
        with open(os.path.join(self.eval_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4, cls=NpEncoder)

    def _get_evaluator(self):
        if self.model == 'cmvae':
            return EvalCMVAE
        elif self.model == 'contrast_crl':
            return EvalContrastCRL
        elif self.model == 'pcl':
            return EvalPCL
