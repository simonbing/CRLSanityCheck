import os
import sys

from absl import flags, app
import torch
import wandb

from crc.baselines.citris.experiments.utils import load_datasets, train_model
from crc.baselines.citris.models.citris_vae.lightning_module import CITRISVAE

FLAGS = flags.FLAGS

# General params
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum('model', 'citrisvae', ['citrisvae'],
                  'Model to train.')
flags.DEFINE_string('data_root', './data/chamber_downloads', 'Root directory where data is saved.')
flags.DEFINE_string('root_dir', './results', 'Root directory where output is saved.')
flags.DEFINE_enum('dataset', 'pong', ['pong', 'chambers', 'chambers_semi_synth_decoder'],
                  'Dataset for training.')
flags.DEFINE_string('task', None, 'Experimental task for training.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')

# Training params
flags.DEFINE_integer('epochs', 200, 'Number of training epochs.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')

# Model params
flags.DEFINE_integer('lat_dim', 5, 'Latent dimension.')
flags.DEFINE_integer('c_hid', 64, 'Hidden dimension of model.')

# wandb params
flags.DEFINE_string('wandb_project', None, 'Name of the wandb project to log experiments.')
flags.DEFINE_string('wandb_username', None, 'Username for wandb logging.')


def main(argv):
    ############### wandb section ###############
    # Can be ignored if not using wandb for experiment tracking
    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size
    )

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace() or None in [FLAGS.wandb_project, FLAGS.wandb_username]:
        print( 'Not using wandb for logging! This could be due to missing project and username flags!')
        wandb_mode = 'offline'
    else:
        wandb_mode = 'online'

    wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_username,
        mode=wandb_mode,
        config=wandb_config
    )
    ##############################################

    # Training preparation
    torch.set_float32_matmul_precision('high')

    datasets, data_loaders, data_name = load_datasets(seed=FLAGS.seed,
                                                      dataset_name=FLAGS.dataset,
                                                      data_dir=FLAGS.data_root,
                                                      seq_len=2,
                                                      batch_size=FLAGS.batch_size,
                                                      num_workers=10 if not gettrace() else 0)

    # Make directories to save results
    model_dir = os.path.join(FLAGS.root_dir, FLAGS.dataset, FLAGS.task, FLAGS.model)
    train_dir = os.path.join(model_dir, FLAGS.run_name, f'seed_{FLAGS.seed}', 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    model_class = CITRISVAE

    img_width = datasets['train'].get_img_width() if FLAGS.dataset == 'pong' else datasets['train'].dataset.get_img_width()
    num_causal_vars = datasets['train'].num_vars() if FLAGS.dataset == 'pong' else datasets['train'].dataset.num_vars()
    c_in = datasets['train'].get_inp_channels() if FLAGS.dataset == 'pong' else datasets['train'].dataset.get_inp_channels()


    model_args = {
        'data_folder': FLAGS.data_root,
        'img_width': img_width,
        'num_causal_vars': num_causal_vars,
        'max_iters': FLAGS.epochs * len(data_loaders['train']),
        'c_in': c_in,  # Nr of input channels
        'batch_size': FLAGS.batch_size,
        'num_workers': 10 if not gettrace() else 0,
        'exclude_vars': None,
        'exclude_objects': None,
        'coarse_vars': False,
        'data_img_width': -1,
        'seq_len': 2,
        'lr': FLAGS.lr,
        'warmup': int(FLAGS.epochs/10),
        'imperfect_interventions': False,
        'c_hid': FLAGS.c_hid,
        'decoder_num_blocks': 1,
        'act_fn': 'silu',
        'num_latents': FLAGS.lat_dim,
        'classifier_lr': 0.004,
        'classifier_momentum': 0.0,
        'classifier_gumbel_temperature': 1.0,
        'classifier_use_normalization': False,
        'classifier_use_conditional_targets': False,
        'kld_warmup': 0,
        'beta_t1': 1.0,
        'gamma': 1.0,
        'lambda_reg': 0.0,
        'autoregressive_prior': False,
        'use_flow_prior': True,
        'beta_classifier': 2.0,
        'beta_mi_estimator': 2.0,
        'lambda_sparse': 0.02,
        'mi_estimator_comparisons': 1,
        'graph_learning_method': 'ENCO'
    }

    # Training (includes evaluation)
    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val'],
                test_loader=data_loaders['test'],
                root_dir=train_dir,
                seed=FLAGS.seed,
                max_epochs=FLAGS.epochs,
                logger_name=f'{FLAGS.model}_{FLAGS.lat_dim}l_{model_args["num_causal_vars"]}b_{FLAGS.c_hid}hid_{data_name}',
                check_val_every_n_epoch=1 if not gettrace() else 2,
                progress_bar_refresh_rate=1 if not gettrace() else 1,
                callback_kwargs={'dataset': datasets['train'],
                                 'correlation_dataset': datasets['val'],  # Independent latents here
                                 'correlation_test_dataset': datasets['test']},
                var_names=datasets['train'].dataset.target_names(),
                causal_var_info=datasets['train'].dataset.get_causal_var_info(),
                save_last_model=True,
                cluster_logging=False,
                **model_args)

    print('Training finished!')


if __name__ == '__main__':
    app.run(main)
