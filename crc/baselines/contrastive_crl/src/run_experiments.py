import copy
import os
from pathlib import Path
import pickle
import argparse
import yaml
import time

import numpy as np
import torch
import pandas as pd

from src.training import train_model
from src.evaluation import check_recovery
from src.models import build_model_from_kwargs
from src.data_generation import get_data_from_kwargs
from src.utils import sanity_checks_kwargs, generate_images


def run_method(databag, directory, model_kwargs, training_kwargs):
    val_loss = np.inf
    best_model = None
    dl_obs, dl_int = databag.get_dataloaders(batch_size=training_kwargs['batch_size'], train=True)
    dl_obs_val, dl_int_val = databag.get_dataloaders(batch_size=training_kwargs['batch_size'], train=False)
    filename_model = os.path.join(directory, 'best_model.pt')
    filename_W = os.path.join(directory, 'W.npy')
    filename_result = os.path.join(directory, 'result.pkl')
    Path(directory).mkdir(parents=True, exist_ok=True)
    if os.path.exists(filename_result):
        with open(filename_result, 'rb') as handle:
            results = pickle.load(handle)
        best_model = torch.load(filename_model)
        result_update = check_recovery(databag, best_model, training_kwargs.get('device', 'cpu'))
        # save_arrays(databag, best_model, directory, training_kwargs.get('device', 'cpu'))
        results.update(result_update)
        with open(filename_result, 'wb') as handle:
            pickle.dump(results, handle)
        return results

    start_time = time.time()
    for i in range(training_kwargs['restarts'] + 1):
        model = build_model_from_kwargs(model_kwargs)
        if model_kwargs['type'] == 'oracle':
            model.embedding.fc = torch.tensor(databag.get_H().T, dtype=torch.float)
        best_model_iter, model, val_loss_iter, loss_info = train_model(model, dl_obs, dl_int, dl_obs_val,
                                                            dl_int_val,
                                                            training_kwargs=training_kwargs,
                                                            z_gt=databag.obs_val[:databag.val_samples],
                                                            x_val=databag.get_observations()
                                                            )
        filename_loss_history = os.path.join(directory, 'loss_history_{}.csv'.format(i))
        create_csv_from_loss(loss_info, filename_loss_history)
        if val_loss_iter <= val_loss:
            val_loss = val_loss_iter
            best_model = best_model_iter
    run_time = time.time() - start_time
    torch.save(best_model, filename_model)
    W = best_model.parametric_part.A.t().cpu().detach().numpy()
    np.save(filename_W, W)
    if hasattr(best_model, 'decoder') and databag.mixing == 'image':
        generate_images(best_model, databag, directory, device=training_kwargs.get('device', 'cpu'))
    results = check_recovery(databag, best_model, training_kwargs.get('device', 'cpu'))
    results['run_time'] = run_time
    with open(filename_result, 'wb') as handle:
        pickle.dump(results, handle)
    return results


def start_single_run(directory, seed, model_kwargs, training_kwargs, data_kwargs):
    data_kwargs, model_kwargs, training_kwargs = sanity_checks_kwargs(data_kwargs, model_kwargs, training_kwargs)
    data_kwargs_copy = copy.deepcopy(data_kwargs)
    data_kwargs_copy['seed'] = seed

    databag = get_data_from_kwargs(data_kwargs_copy)
    model_path = os.path.join(directory, 'model_f.pt')
    w_path = os.path.join(directory, 'W.npy')
    results = dict()
    if os.path.exists(model_path):
        databag.f = torch.load(model_path)
    elif data_kwargs_copy['mixing'] != 'image':
        torch.save(databag.f, model_path)
    if os.path.exists(w_path):
        databag.W = np.load(w_path)
    else:
        np.save(w_path, databag.W)
    # for type in ['vae', 'contrastive']:
    model_types = ['contrastive', 'vae', 'vae_vanilla', 'contrastive_linear']
    if data_kwargs_copy['mixing'] == 'image':
        model_types = ['contrastive', 'vae', 'vae_vanilla', 'vae_contrastive']
    for type in model_types:
        print('Running method {}'.format(type))
        model_kwargs_copy = copy.deepcopy(model_kwargs)
        model_kwargs_copy['type'] = type

        training_kwargs_copy = copy.deepcopy(training_kwargs)
        training_kwargs_copy['type'] = type
        directory_method = os.path.join(directory, type)
        results[type] = run_method(databag, directory_method, model_kwargs_copy, training_kwargs_copy)
        print(results[type])
    if training_kwargs['run_baseline']:
        from baselines.run_linear import run_linear_disentanglement
        try:
            results_baseline, W = run_linear_disentanglement(databag)
        except np.linalg.LinAlgError:
            print('Matrix in baseline method is not positive definite')
            results_baseline, W = None, None
        directory_method = os.path.join(directory, 'linear_baseline')
        Path(directory_method).mkdir(parents=True, exist_ok=True)
        if W is not None:
            np.save(os.path.join(directory_method, 'W.npy'), W)
            results['linear_baseline'] = results_baseline
            with open(os.path.join(directory_method, 'result.pkl'), 'wb') as handle:
                pickle.dump(results_baseline, handle)
            print(results_baseline)
    return results


def start_experiment(directory, index=-1):
    with open(os.path.join(directory, 'settings.yaml'), "r") as stream:
        params = yaml.safe_load(stream)

    runs = params['data']['runs']
    seed = params['data']['seed']
    results = []
    if index < 0:
        jobs = range(runs)
    elif index < runs:
        jobs = [index]
    else:
        return
    for i in jobs:
        dir_run = os.path.join(directory, str(i).zfill(3))
        Path(dir_run).mkdir(parents=True, exist_ok=True)
        results_run = start_single_run(dir_run, seed + i, model_kwargs=params['model'], training_kwargs=params['train'],
                                       data_kwargs=params['data'])
        results.append(results_run)

    create_csv_from_results(results, filename=os.path.join(directory, 'results.csv'))


def create_csv_from_results(results, filename):
    lines = []
    for i, res in enumerate(results):
        for method, method_results in res.items():
            method_results['method'] = method
            method_results['run'] = i
            lines.append(method_results)
    df = pd.DataFrame.from_records(lines)
    df.to_csv(filename, index=False)


def create_csv_from_loss(results, filename):
    df = pd.DataFrame.from_records(results).T
    df.columns = ['train_loss', 'val_loss', 'mean_r2']
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", required=True, help="Base save directory for the evaluation"
    )
    parser.add_argument("--run_subdirs", action='store_true')
    parser.add_argument("--index", default=-1, type=int)

    args = parser.parse_args()
    if args.run_subdirs:
        list_dir = os.listdir(args.train_dir)
        for dir in list_dir:
            dir_path = os.path.join(args.train_dir, dir)
            if os.path.isdir(dir_path):
                start_experiment(dir_path)
    else:
        start_experiment(args.train_dir, args.index)
