

import copy
import os
from pathlib import Path
import pickle
import argparse

import pandas.errors
import yaml
import time

import numpy as np
import torch
import pandas as pd

from src.evaluation import evaluate_graph_metrics


def evaluate_single_run(directory, run_nr, settings):
    subfolders = [(f.path, f.name) for f in os.scandir(directory) if f.is_dir()]
    results = []
    filename_W_true = os.path.join(directory, 'W.npy')
    if os.path.exists(filename_W_true):
        W_true = np.load(os.path.join(directory, 'W.npy'))
    else:
        return results
    for dirs in subfolders:
        result_filename = os.path.join(dirs[0], 'result.pkl')
        w_filename = os.path.join(dirs[0], 'W.npy')
        if os.path.exists(result_filename):
            with open(result_filename, 'rb') as f:
                res = pickle.load(f)
            res['run'] = run_nr
            res['method'] = dirs[1]
            if os.path.exists(w_filename):
                W = np.load(w_filename)
                res_upt = evaluate_graph_metrics(W_true, W, 1e-9, (settings['data']['k'] * settings['data']['d']) // 2)
                res.update(res_upt)
            results.append(res)
    return results


def evaluate_experiment(directory, group):
    print('Evaluating directory {}'.format(directory))
    subfolders = [(f.path, f.name) for f in os.scandir(directory) if f.is_dir()]
    filename = os.path.join(directory, 'results.csv')

    with open(os.path.join(directory, 'settings.yaml'), "r") as stream:
        settings = yaml.safe_load(stream)
    results = []
    for dirs in subfolders:
        results = results + evaluate_single_run(dirs[0], int(dirs[1]), settings)
    df = pd.DataFrame.from_records(results)
    df.to_csv(filename, index=False)
    # print(df)
    if len(df) != 0 and (group is None or settings['data'].get('name', None) == group):
        keys = ['method', 'SHD', 'FDR', 'TPR', 'FPR', 'R2_mean', 'SHD_opt', 'FPR_opt', 'TPR_opt', 'FDR_opt', 'mcc_s_out', 'opt_thresh']
        keys = ['method', 'SHD', 'FDR', 'TPR', 'FPR', 'R2_mean', 'SHD_edge_matched', 'auroc', 'R2_sq_mean', 'mcc_s_out', 'opt_thresh']
        keys_red = [k for k in keys if k in df.columns]
        df = df[keys_red]
        df_means = df.groupby(['method']).mean()
        df_means['counts'] = df.groupby(['method']).agg({'method': 'size'})
        print(df_means.to_string())


def aggregate_all_experiments(dir_base):
    list_dir = os.listdir(dir_base)
    results = []
    for dir in list_dir:
        dir_path = os.path.join(args.run_dir, dir)
        results_path = os.path.join(dir_path, 'results.csv')
        if os.path.exists(results_path):
            settings_path = os.path.join(dir_path, 'settings.yaml')

            with open(settings_path, "r") as stream:
                settings = yaml.safe_load(stream)
            try:
                df = pd.read_csv(results_path)
                for domain in settings.keys():
                    for key in settings[domain].keys():
                        if isinstance(settings[domain][key], list):
                            df[key+'_lower'] = settings[domain][key][0]
                            df[key+'_upper'] = settings[domain][key][1]
                        else:
                            df[key] = settings[domain][key]
                            df[key+'_lower'] = settings[domain][key]
                            df[key+'_upper'] = settings[domain][key]
                results.append(df)
            except:
                pandas.errors.EmptyDataError
    df = pd.concat(results)
    df.to_csv(os.path.join(dir_base, 'all_results.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", required=True, help="Base save directory for the evaluation"
    )
    parser.add_argument("--run_subdirs", action='store_true')
    parser.add_argument("--select_group", default=None)

    args = parser.parse_args()
    if args.run_subdirs:
        list_dir = os.listdir(args.run_dir)
        for dir in list_dir:
            dir_path = os.path.join(args.run_dir, dir)
            if os.path.isdir(dir_path):
                evaluate_experiment(dir_path, args.select_group)
        aggregate_all_experiments(args.run_dir)
    else:
        evaluate_experiment(args.run_dir)
