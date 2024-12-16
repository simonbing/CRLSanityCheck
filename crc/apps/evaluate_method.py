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
        # Load test dataset
        test_data_path = os.path.join(self.model_dir, 'test_dataset.pkl')
        with open(test_data_path, 'rb') as f:
            test_dataset = pickle.load(f)

        # Get embeddings from self.method
        z, z_hat = self.method.get_encodings(test_dataset)
        print(type(z))
        print(type(z_hat))
        print(type(np.asarray(z)))
        print(isinstance(z, np.ndarray))
        print(isinstance(np.asarray(z), np.ndarray))

        # Calculate metrics
        results = {}
        if 'mcc' in self.metrics:
            mmc_dict = compute_mccs(z, z_hat)
            results['mcc'] = mmc_dict['mcc_s_out']
            results['mcc_lin'] = mmc_dict['mcc_w_out']
        if 'shd' in self.metrics:
            try:
                W_gt = test_dataset.dataset.W
                nr_edges = np.count_nonzero(W_gt)
                shd_dict = evaluate_graph_metrics(
                    W_gt,
                    self.method.model.A.t().cpu().detach().numpy(),
                    nr_edges=nr_edges)
                results['shd'] = shd_dict['SHD']
                results['shd_opt'] = shd_dict['SHD_opt']
                results['shd_edge_match'] = shd_dict['SHD_edge_matched']
            finally:
                pass

        # Log results (before concatenation)
        for key in results:
            wandb.run.summary[key] = results[key]

        # Concat all results dicts, save as json
        results_cat = results | mmc_dict
        results_path = os.path.join(self.eval_dir, 'results.json')
        with open(results_path, 'w') as outfile:
            json.dump(results_cat, outfile, indent=4, cls=NpEncoder)
