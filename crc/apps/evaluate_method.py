import json
import os
import pickle
import random

import numpy as np
import torch
import wandb

from crc.utils import NpEncoder
from crc.methods.utils import get_method
from crc.baselines.contrastive_crl.src.evaluation import compute_mccs, evaluate_graph_metrics
from crc.eval import compute_multiview_r2


class EvaluateMethod(object):
    def __init__(self, method, out_dir, run_name, metrics, trained_method=None, **kwargs):
        self.method_name = method
        self.model_dir = os.path.join(out_dir, kwargs['dataset'],
                                      kwargs['task'],
                                      self.method_name)
        self.eval_dir = os.path.join(self.model_dir, run_name,
                                     f"seed_{kwargs['seed']}", 'eval')
        trained_model_path = os.path.join(self.model_dir, run_name,
                                          f"seed_{kwargs['seed']}", 'train',
                                          'best_model.pt')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        if trained_method is None:  # Get method
            self.method = get_method(method=method)(**kwargs)
        else:
            self.method = trained_method

        # Load best model from training
        self.method.model = torch.load(trained_model_path)

        self.metrics = metrics

    def run(self):
        # Load test dataset_name
        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        with open(test_data_path, 'rb') as f:
            test_dataset = pickle.load(f)

        # Get embeddings from self.method
        z, z_hat = self.method.get_encodings(test_dataset)

        # Calculate metrics
        results = {}
        if 'mcc' in self.metrics:
            mcc_dict = compute_mccs(z, z_hat)
            results['mcc'] = mcc_dict['mcc_s_out']
            results['mcc_lin'] = mcc_dict['mcc_w_out']
        if 'shd' in self.metrics:
            try:
                W_gt = test_dataset.dataset_name.W
                W_hat = np.asarray(self.method.model.A.t().cpu().detach().numpy())
                nr_edges = np.count_nonzero(W_gt)
                shd_dict = evaluate_graph_metrics(
                    W_gt,
                    W_hat,
                    nr_edges=nr_edges)
                results['shd'] = shd_dict['SHD']
                results['shd_opt'] = shd_dict['SHD_opt']
                results['shd_edge_match'] = shd_dict['SHD_edge_matched']
            finally:
                pass
        if 'r2' in self.metrics:  # Block identifiability
            r2_dict = compute_multiview_r2(z, z_hat, test_dataset.dataset.content_indices,
                                 test_dataset.dataset.subsets)
            results['avg_r2_lin'] = r2_dict['avg_r2_lin']
            results['avg_r2_nonlin'] = r2_dict['avg_r2_nonlin']

            # Save r2 results dict
            r2_dict_path = os.path.join(self.eval_dir, 'r2_dict.pkl')
            with open(r2_dict_path, 'wb') as f:
                pickle.dump(r2_dict, f)


        # Log results (before concatenation)
        for key in results:
            wandb.run.summary[key] = results[key]

        # Concat all results dicts, save as json
        try:
            results = results | mcc_dict
        except UnboundLocalError:
            pass

        results_path = os.path.join(self.eval_dir, 'results.json')
        with open(results_path, 'w') as outfile:
            json.dump(results, outfile, indent=4, cls=NpEncoder)
