import json
import os
import pickle
import random
import sys

from absl import flags, app
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from crc.utils import get_device, NpEncoder
from crc.baselines.contrastive_crl.src.utils import get_chamber_data
from crc.baselines.contrastive_crl.src.models import get_contrastive_synthetic, \
    get_contrastive_image
from crc.baselines.contrastive_crl.src.training import train_model
from crc.baselines.contrastive_crl.src.evaluation import compute_mccs, evaluate_graph_metrics
from crc.baselines.contrastive_crl.src.data_generation import ContrastiveCRLDataset

FLAGS = flags.FLAGS

# General params
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum('model', 'contrast_crl', ['contrast_crl', 'contrast_crl_linear'],
                  'Model to train.')
flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/'
                                 'CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory where output is saved.')
flags.DEFINE_enum('dataset', None, ['lt_camera_v1', 'lt_camera_walks_v1',
                                    'lt_crl_benchmark_v1',
                                    'contrast_synth', 'contrast_img',
                                    'contrast_semi_synth_mlp', 'contrast_synth_re',
                                    'contrast_semi_synth_decoder'],
                  'Dataset for training.')
flags.DEFINE_string('task', None, 'Experimental task for training.')
flags.DEFINE_bool('overwrite_data', False, 'Overwrite existing saved data.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')

# Training params
flags.DEFINE_enum('optim', 'adam', ['adam', 'sgd'], 'Optimizer for training.')
flags.DEFINE_float('lr', 0.0005, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')

# Model params
flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')
flags.DEFINE_float('mu', 0.00001, 'Mu hyperparam.')
flags.DEFINE_float('eta', 0.0001, 'Eta hyperparam.')
flags.DEFINE_float('kappa', 0.1, 'Kappa hyperparam.')


def main(argv):
    # wandb stuff
    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs,
        lat_dim=FLAGS.lat_dim
    )

    gettrace = getattr(sys, 'gettrace', None)

    wandb.init(
        project='chambers',
        entity='CausalRepresentationChambers',  # this is the team name in wandb
        mode='online' if not gettrace() else 'offline',
        # don't log if debugging
        config=wandb_config
    )

    # Training preparation
    torch.set_float32_matmul_precision('high')

    # Set all seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Make directories to save results
    model_dir = os.path.join(FLAGS.root_dir, FLAGS.dataset, FLAGS.task, FLAGS.model)
    train_dir = os.path.join(model_dir, FLAGS.run_name, f'seed_{FLAGS.seed}', 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    device = get_device()
    print(f'using device: {device}')

    # Training
    # Check if trained model already exists, skip training if so
    if os.path.exists(os.path.join(train_dir, 'best_model.pt')):
        print('Trained model found, skipping training!')
        pass
    else:
        # Get data
        dataset_train, dataset_val, dataset_test = get_chamber_data(
            dataset=FLAGS.dataset,
            task=FLAGS.task,
            data_root=FLAGS.data_root,
            seed=FLAGS.seed)

        # Save train data (as torch dataset)
        if FLAGS.dataset == 'contrast_synth':
            train_data_path = os.path.join(model_dir,
                                           f'train_dataset_seed_{FLAGS.seed}.pkl')
        else:
            train_data_path = os.path.join(model_dir, 'train_dataset.pkl')
        if not os.path.exists(train_data_path) or FLAGS.overwrite_data:
            with open(train_data_path, 'wb') as f:
                pickle.dump(dataset_train, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save val data
        if FLAGS.dataset == 'contrast_synth':
            val_data_path = os.path.join(model_dir,
                                         f'val_dataset_seed_{FLAGS.seed}.pkl')
        else:
            val_data_path = os.path.join(model_dir, 'val_dataset.pkl')
        if not os.path.exists(val_data_path) or FLAGS.overwrite_data:
            with open(val_data_path, 'wb') as f:
                pickle.dump(dataset_val, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save test data
        if FLAGS.dataset == 'contrast_synth':
            test_data_path = os.path.join(model_dir,
                                          f'test_dataset_seed_{FLAGS.seed}.pkl')
        else:
            test_data_path = os.path.join(model_dir, 'test_dataset.pkl')
        if not os.path.exists(test_data_path) or FLAGS.overwrite_data:
            with open(test_data_path, 'wb') as f:
                pickle.dump(dataset_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Make dataloaders
        dl_train = DataLoader(dataset_train, shuffle=True,
                              batch_size=FLAGS.batch_size,
                              num_workers=24 if not gettrace() else 0)
        dl_val = DataLoader(dataset_val, shuffle=False,
                            batch_size=FLAGS.batch_size,
                            num_workers=24 if not gettrace() else 0)

        # Build model
        match FLAGS.dataset:
            case a if a in ('contrast_synth', 'contrast_semi_synth_mlp', 'contrast_synth_re'):
                model = get_contrastive_synthetic(input_dim=20,
                                                  latent_dim=FLAGS.lat_dim,
                                                  hidden_dim=512,
                                                  hidden_layers=0,
                                                  residual=True)
            case b if b in ('lt_crl_benchmark_v1', 'contrast_semi_synth_decoder'):
                model = get_contrastive_image(latent_dim=FLAGS.lat_dim,
                                              conv=True,
                                              channels=10)  # this is not used
            case _:
                model = get_contrastive_image(latent_dim=FLAGS.lat_dim,
                                              conv=False,  # TODO: figure out this weird variable!
                                              channels=10)

        training_kwargs = {
            'epochs': FLAGS.epochs,
            'optimizer': FLAGS.optim,
            'mu': FLAGS.mu,
            'eta': FLAGS.eta,
            'kappa': FLAGS.kappa,
            'lr_nonparametric': FLAGS.lr,
            'weight_decay': 0.0
        }

        # Train model
        best_model, last_model, _, _ = train_model(model, device, dl_train,
                                                   dl_val, training_kwargs,
                                                   verbose=True)
        # Save model
        torch.save(best_model, os.path.join(train_dir, 'best_model.pt'))
        torch.save(last_model, os.path.join(train_dir, 'last_model.pt'))

        print('Training finished!')

    # Evaluation
    eval_dir = os.path.join(model_dir, FLAGS.run_name, f'seed_{FLAGS.seed}', 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Load test data (all ground truth info should be here!)
    if FLAGS.dataset == 'contrast_synth':
        dataset_test_path = os.path.join(model_dir, f'test_dataset_seed_{FLAGS.seed}.pkl')
    else:
        dataset_test_path = os.path.join(model_dir, 'test_dataset.pkl')
    with open(dataset_test_path, 'rb') as f:
        dataset_test = pickle.load(f)

    results = {}

    # Get trained model
    trained_model = torch.load(os.path.join(train_dir, 'best_model.pt'))
    trained_model = trained_model.to(device)

    # Get encodings
    trained_model.eval()

    if isinstance(dataset_test, ContrastiveCRLDataset):
        z_gt = dataset_test.z_obs.cpu().detach().numpy()
        x_gt = dataset_test.f(torch.tensor(z_gt, dtype=torch.float)).to(device)

        z_hat = trained_model.get_z(x_gt).cpu().detach().numpy()
    else:
        dataset_test.dataset.eval = True
        dataloader_test = DataLoader(dataset_test, batch_size=2000, shuffle=False)

        z_list = []
        z_hat_list = []

        # Iterate over test dataloader and encode all samples and save gt data
        for X in dataloader_test:
            x_obs = X[0]
            z_obs = X[-1]

            x_obs = x_obs.to(device)

            z_hat_batch = trained_model.get_z(x_obs)

            z_list.append(z_obs)
            z_hat_list.append(z_hat_batch)

        z_gt = torch.cat(z_list).cpu().detach().numpy()
        z_hat = torch.cat(z_hat_list).cpu().detach().numpy()

    z_gt = np.asarray(z_gt, dtype=np.float32)
    z_hat = np.asarray(z_hat, dtype=np.float32)

    # MCC metric
    z_pred_sign_matched = z_hat * np.sign(z_hat)[:, 0:1] * np.sign(z_gt)[:, 0:1]

    mccs = compute_mccs(z_gt, z_hat)
    mccs_sign_matched = compute_mccs(z_gt, z_pred_sign_matched)
    mccs_abs = compute_mccs(np.abs(z_gt), np.abs(z_hat))

    results['mcc_in'] = mccs['mcc_s_in']
    results['mcc_out'] = mccs['mcc_s_out']
    results['mcc_w_in'] = mccs['mcc_w_in']
    results['mcc_w_out'] = mccs['mcc_w_out']

    print('Alternative MCC scores:')
    print(mccs_sign_matched)
    print(mccs_abs)

    # SHD metric
    try:
        G_gt = dataset_test.W
    except AttributeError:
        G_gt = dataset_test.dataset.W

    G_hat = trained_model.parametric_part.A.t().cpu().detach().numpy()

    G_gt = np.asarray(G_gt, dtype=np.float32)
    G_hat = np.asarray(G_hat, dtype=np.float32)

    nr_edges = np.count_nonzero(G_gt)
    shd_dict = evaluate_graph_metrics(G_gt, G_hat, nr_edges=nr_edges)

    results = results | shd_dict

    # Log to wandb summary
    for key in results:
        wandb.run.summary[key] = results[key]

    # Save results
    with open(os.path.join(eval_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, cls=NpEncoder)

    print('Evaluation finished!')


if __name__ == '__main__':
    app.run(main)
