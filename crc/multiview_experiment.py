import copy
import csv
import os
import pickle
import random
import sys

from absl import flags, app
# import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18
import wandb

from crc.utils import get_device, train_val_test_split
from crc.shared.architectures import FCEncoder
from crc.baselines.contrastive_crl.src.models import EmbeddingNet
from crc.baselines.multiview_crl.invertible_network_utils import construct_invertible_mlp
from crc.baselines.multiview_crl.datasets import Multimodal3DIdent, MultiviewSynthDataset
from crc.baselines.multiview_crl.encoders import TextEncoder2D
from crc.baselines.multiview_crl.infinite_iterator import InfiniteIterator
from crc.baselines.multiview_crl.losses import infonce_loss
from crc.baselines.multiview_crl.main_multimodal import train_step, get_data, eval_step
import crc.baselines.multiview_crl.dci as dci

FLAGS = flags.FLAGS

# General params
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum('model', 'multiview_crl', ['multiview_crl'],
                  'Model to train.')
flags.DEFINE_string('data_root', '/Users/Simon/Documents/PhD/Projects/'
                                 'CausalRepresentationChambers/data/chamber_downloads',
                    'Root directory where data is saved.')
flags.DEFINE_string('root_dir', '/Users/Simon/Documents/PhD/Projects/CausalRepresentationChambers/results',
                    'Root directory where output is saved.')
flags.DEFINE_enum('dataset', 'multimodal3di', ['multimodal3di', 'multiview_synth'],
                  'Dataset for training.')
flags.DEFINE_string('task', None, 'Experimental task for training.')
flags.DEFINE_bool('overwrite_data', False, 'Overwrite existing saved data.')
flags.DEFINE_string('run_name', None, 'Name for the training run.')

# Training params
# flags.DEFINE_enum('optim', 'adam', ['adam', 'sgd'], 'Optimizer for training.')
flags.DEFINE_float('lr', 0.0005, 'Learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('train_steps', 100000, 'Training steps.')
flags.DEFINE_integer('checkpoint_steps', 1000, 'Checkpoint interval steps.')
flags.DEFINE_integer('log_steps', 100, 'Log interval steps.')
flags.DEFINE_bool('save_all_checkpoints', False, 'Save all checkpoints.')
# flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_float('tau', 1.0, 'Temperature parameter for multiview loss.')

# Model params
flags.DEFINE_integer('lat_dim', 11, 'Latent dimension.')

# Evaluation params
flags.DEFINE_bool('eval_dci', False, 'Evaluate DCI metric.')
flags.DEFINE_bool('eval_style', False, 'Evaluate style variables, too.')


def get_datasets(dataset):
    match dataset:
        case 'multimodal3di':
            # Normalization of the input images; (r, g, b) channels
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.4327, 0.2689, 0.2839],
                                         [0.1201, 0.1457, 0.1082]),
                ]
            )
            change_lists = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

            train_dataset = Multimodal3DIdent(
                data_dir=FLAGS.data_root,
                mode='train',
                change_lists=change_lists,
                transform=transform
            )
            val_dataset = Multimodal3DIdent(
                data_dir=FLAGS.data_root,
                mode='val',
                change_lists=change_lists,
                transform=transform,
                vocab_filepath=train_dataset.vocab_filepath
            )
            test_dataset = Multimodal3DIdent(
                data_dir=FLAGS.data_root,
                mode='test',
                change_lists=change_lists,
                transform=transform,
                vocab_filepath=train_dataset.vocab_filepath
            )
        case 'multiview_synth':
            full_dataset = MultiviewSynthDataset(
                n=145000,
                # transforms=[
                #     construct_invertible_mlp(
                #         n=3,
                #         n_layers=3,
                #         n_iter_cond_thresh=25000,
                #         cond_thresh_ratio=0.001),
                #     construct_invertible_mlp(
                #         n=3,
                #         n_layers=3,
                #         n_iter_cond_thresh=25000,
                #         cond_thresh_ratio=0.001),
                #     construct_invertible_mlp(
                #         n=3,
                #         n_layers=3,
                #         n_iter_cond_thresh=25000,
                #         cond_thresh_ratio=0.001),
                #     construct_invertible_mlp(
                #         n=3,
                #         n_layers=3,
                #         n_iter_cond_thresh=25000,
                #         cond_thresh_ratio=0.001)
                # ]
                # Alternative with encoders scaling up to larger dimension
                transforms=[
                    EmbeddingNet(3, 20, 512, hidden_layers=3, residual=False),
                    EmbeddingNet(3, 20, 512, hidden_layers=3, residual=False),
                    EmbeddingNet(3, 20, 512, hidden_layers=3, residual=False),
                    EmbeddingNet(3, 20, 512, hidden_layers=3, residual=False)
                ]
            )
            # Split data
            train_frac = 0.862
            val_frac = 0.069
            test_frac = 0.069

            train_idxs, val_idxs, test_idxs = \
                train_val_test_split(list(range(len(full_dataset))),
                                     train_size=train_frac, val_size=val_frac,
                                     random_state=42)

            train_dataset = Subset(full_dataset, train_idxs)
            val_dataset = Subset(full_dataset, val_idxs)
            test_dataset = Subset(full_dataset, test_idxs)

    return train_dataset, val_dataset, test_dataset


def get_encoders(dataset):
    match dataset:
        case 'multimodal3di':
            # Define image encoder
            encoder_img = torch.nn.Sequential(
                resnet18(num_classes=100),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(100, 11),
            )
            encoder_img = torch.nn.DataParallel(encoder_img, device_ids=[0])

            # Define text encoder
            sequence_length = 26
            encoder_txt = TextEncoder2D(
                input_size=111,
                output_size=11,
                sequence_length=sequence_length,
            )
            encoder_txt = torch.nn.DataParallel(encoder_txt, device_ids=[0])

            encoders = [encoder_img, encoder_txt]
        case 'multiview_synth':
            # encoder_1 = FCEncoder(in_dim=3, latent_dim=4,
            #                       hidden_dims=[64, 256, 256, 256, 64])
            # encoder_2 = FCEncoder(in_dim=3, latent_dim=4,
            #                       hidden_dims=[64, 256, 256, 256, 64])
            # encoder_3 = FCEncoder(in_dim=3, latent_dim=4,
            #                       hidden_dims=[64, 256, 256, 256, 64])
            # encoder_4 = FCEncoder(in_dim=3, latent_dim=4,
            #                       hidden_dims=[64, 256, 256, 256, 64])
            # Alernative for other mixing (larger in dim)
            encoder_1 = FCEncoder(in_dim=20, latent_dim=4,
                                  hidden_dims=[64, 256, 256, 256, 64])
            encoder_2 = FCEncoder(in_dim=20, latent_dim=4,
                                  hidden_dims=[64, 256, 256, 256, 64])
            encoder_3 = FCEncoder(in_dim=20, latent_dim=4,
                                  hidden_dims=[64, 256, 256, 256, 64])
            encoder_4 = FCEncoder(in_dim=20, latent_dim=4,
                                  hidden_dims=[64, 256, 256, 256, 64])

            encoders = [encoder_1, encoder_2, encoder_3, encoder_4]

    return encoders


def get_train_args(dataset):
    match dataset:
        case 'multimodal3di':
            modalities = ['image', 'text']
            content_indices = [[0, 10], [0, 1, 2], [0], [0]]
            subsets = [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
            n_views = 3
            style_indices = [3, 4, 5, 6, 7, 8, 9]  # Anything that is not a content var
        case 'multiview_synth':
            modalities = ['view_1', 'view_2', 'view_3', 'view_4']
            content_indices = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3],
                               [0], [1], [2], [3]]
            subsets = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                       (0, 1, 2), (0, 1, 3,), (0, 2, 3), (1, 2, 3)]
            n_views = 4
            style_indices = []

    return modalities, content_indices, subsets, n_views, style_indices


def loss_func(z_rec_tuple, estimated_content_indices, subsets):
    return infonce_loss(
        z_rec_tuple,
        sim_metric=torch.nn.CosineSimilarity(dim=-1),
        criterion=torch.nn.CrossEntropyLoss(),
        tau=FLAGS.tau,
        projector=(lambda x: x),
        # invertible_network_utils.construct_invertible_mlp(n=args.encoding_size, n_layers=2).to(device),
        estimated_content_indices=estimated_content_indices,
        subsets=subsets,
    )


def main(argv):
    # wandb stuff
    wandb_config = dict(
        model=FLAGS.model,
        dataset=FLAGS.dataset,
        task=FLAGS.task,
        run_name=FLAGS.run_name,
        seed=FLAGS.seed,
        batch_size=FLAGS.batch_size,
        train_steps=FLAGS.train_steps,
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
    # Set all seeds
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # faiss.omp_set_num_threads(16)

    # Make directories to save results
    model_dir = os.path.join(FLAGS.root_dir, FLAGS.dataset, FLAGS.task, FLAGS.model)
    train_dir = os.path.join(model_dir, FLAGS.run_name, f'seed_{FLAGS.seed}', 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    device = get_device()
    print(f'using device: {device}')

    # Training
    # Get training args
    modalities, content_indices, subsets, n_views, style_indices = get_train_args(
        FLAGS.dataset)
    if all([os.path.exists(os.path.join(train_dir, f'encoder_{m}.pt'))
            for m in modalities]):
        print('Trained model found, skipping training!')
        pass
    else:
        train_data_path = os.path.join(model_dir, 'train_dataset.pkl')
        val_data_path = os.path.join(model_dir, 'val_dataset.pkl')
        test_data_path = os.path.join(model_dir, 'test_dataset.pkl')
        if not os.path.exists(train_data_path) or FLAGS.overwrite_data:
            train_dataset, val_dataset, test_dataset = get_datasets(FLAGS.dataset)

            if not os.path.exists(train_data_path) or FLAGS.overwrite_data:
                with open(train_data_path, 'wb') as f:
                    pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            if not os.path.exists(val_data_path) or FLAGS.overwrite_data:
                with open(val_data_path, 'wb') as f:
                    pickle.dump(val_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            if not os.path.exists(test_data_path) or FLAGS.overwrite_data:
                with open(test_data_path, 'wb') as f:
                    pickle.dump(test_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Load data
            with open(train_data_path, 'rb') as f:
                train_dataset = pickle.load(f)

        dataloader_kwargs = {
            "batch_size": FLAGS.batch_size,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 24,
            "pin_memory": True,
        }

        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

        # Get encoders for different views
        encoders = get_encoders(FLAGS.dataset)
        encoders = [encoder.to(device) for encoder in encoders]

        # Define the optimizer
        params = []
        for f in encoders:
            params += list(f.parameters())
        optimizer = torch.optim.Adam(params, lr=FLAGS.lr)

        step = 1
        loss_values = []  # list to keep track of loss values
        while step <= FLAGS.train_steps:
            # training step
            data = next(train_iterator)  # contains images, texts, and labels
            loss_value, _ = train_step(data, device, encoders, loss_func, optimizer, params,
                                       modalities=modalities,
                                       content_indices=content_indices,
                                       subsets=subsets,
                                       n_views_arg=n_views,
                                       selection='ground_truth')
            loss_values.append(loss_value)

            # wandb logging
            wandb.log({'loss': loss_value})

            # save models and intermediate checkpoints
            if step % FLAGS.checkpoint_steps == 1 or step == FLAGS.train_steps or step == FLAGS.log_steps * 2:
                for m_idx, m in enumerate(modalities):
                    torch.save(
                        copy.deepcopy(encoders[m_idx]),
                        os.path.join(train_dir, f"encoder_{m}.pt"),
                    )

                if FLAGS.save_all_checkpoints:
                    torch.save(
                        copy.deepcopy(encoders[m_idx]),
                        os.path.join(train_dir, f"encoder_{m}_%d.pt" % step),
                    )
            step += 1

        print('Training finished!')

    # Evaluation
    eval_dir = os.path.join(model_dir, FLAGS.run_name, f'seed_{FLAGS.seed}',
                            'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Load data
    val_data_path = os.path.join(model_dir, 'val_dataset.pkl')
    test_data_path = os.path.join(model_dir, 'test_dataset.pkl')
    with open(val_data_path, 'rb') as f:
        val_dataset = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)

    # Get trained encoders
    try:
        encoders
    except NameError:
        encoders = [torch.load(os.path.join(train_dir, f'encoder_{m}.pt'))
                    for m in modalities]
        encoders = [encoder.to(device) for encoder in encoders]

    # collect encodings and labels from the validation and test data
    try:
        factors = test_dataset.FACTORS
        discrete_factors = test_dataset.DISCRETE_FACTORS
    except:
        factors = test_dataset.dataset.FACTORS
        discrete_factors = test_dataset.dataset.DISCRETE_FACTORS

    dataloader_kwargs = {
        "batch_size": FLAGS.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 24,
        "pin_memory": True,
    }

    val_dict = get_data(
        val_dataset,
        device,
        encoders,
        loss_func,
        dataloader_kwargs,
        modalities=modalities,
        factors=factors,
        content_indices=content_indices,
        subsets=subsets,
        n_views_arg=n_views,
        selection='ground_truth',
        num_samples=10000,
    )
    test_dict = get_data(
        test_dataset,
        device,
        encoders,
        loss_func,
        dataloader_kwargs,
        modalities=modalities,
        factors=factors,
        content_indices=content_indices,
        subsets=subsets,
        n_views_arg=n_views,
        selection='ground_truth',
        num_samples=10000,
    )

    # standardize the encodings
    for m in modalities:
        scaler = StandardScaler()
        val_dict[f"hz_{m}"] = scaler.fit_transform(val_dict[f"hz_{m}"])
        test_dict[f"hz_{m}"] = scaler.transform(test_dict[f"hz_{m}"])
        for s in subsets:
            scaler = StandardScaler()
            val_dict[f"hz_{m}_subsets"][s] = scaler.fit_transform(
                val_dict[f"hz_{m}_subsets"][s])
            test_dict[f"hz_{m}_subsets"][s] = scaler.transform(
                test_dict[f"hz_{m}_subsets"][s])

    results = []
    for m_idx, m in enumerate(modalities):
        factors_m = factors[m]
        discrete_factors_m = discrete_factors[m]

        if FLAGS.eval_dci:
            # compute dci scores
            def repr_fn(samples):
                f = encoders[m_idx]
                # imgs: np array: [bs, 64, 64, 3]; text: ...
                if m == "image" and FLAGS.dataset == "mpi3d":
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.4327, 0.2689, 0.2839],
                                                 [0.1201, 0.1457, 0.1082]),
                        ]
                    )
                    samples = torch.stack([transform(i) for i in samples],
                                          dim=0)
                with torch.no_grad():
                    hz = f(samples)
                return hz.cpu().numpy()

            # compute DCI scores
            dci_score = dci.compute_dci(
                ground_truth_data=val_dataset,
                representation_function=repr_fn,
                num_train=10000,
                num_test=5000,
                random_state=np.random.RandomState(),
                factor_types=[
                    "discrete" if ix in discrete_factors_m else "continuous" for
                    ix in factors_m],
            )
            # Open the CSV file with write permission
            with open(os.path.join(eval_dir, f"dci_{m}.csv"), "w",
                      newline="") as csvfile:
                # Create a CSV writer using the field/column names
                writer = csv.DictWriter(csvfile, fieldnames=dci_score.keys())
                # Write the header row (column names)
                writer.writeheader()
                # Write the data
                writer.writerow(dci_score)
            continue

        # Standard evaluation begins here
        for ix, factor_name in factors_m.items():
            for s in subsets:
                # select data
                train_inputs = val_dict[f"hz_{m}_subsets"][s]
                test_inputs = test_dict[f"hz_{m}_subsets"][s]
                train_labels = val_dict[f"labels_{m}"][factor_name]
                test_labels = test_dict[f"labels_{m}"][factor_name]
                data = [train_inputs, train_labels, test_inputs, test_labels]

                # append results
                results.append(eval_step(ix, s, m, factor_name, discrete_factors_m, data))
            # independent component extraction
            if FLAGS.eval_style and len(style_indices) > 0:
                # select data
                train_inputs = val_dict[f"hz_{m}"][..., style_indices]
                test_inputs = test_dict[f"hz_{m}"][..., style_indices]
                train_labels = val_dict[f"labels_{m}"][factor_name]
                test_labels = test_dict[f"labels_{m}"][factor_name]
                data = [train_inputs, train_labels, test_inputs, test_labels]
                # append results
                results.append(
                    eval_step(ix, (-1), m, factor_name, discrete_factors_m,
                              data))

    # convert evaluation results into tabular form
    columns = [
        "subset",
        "ix",
        "modality",
        "factor_name",
        "factor_type",
        "r2_linreg",
        "r2_krreg",
        "acc_logreg",
        "acc_mlp",
    ]
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv(os.path.join(eval_dir, "results.csv"))

    # Log final table to wandb
    results_table = wandb.Artifact('results_artifact', type='results')
    results_table.add(wandb.Table(dataframe=df_results), 'results_table')

    print(df_results.to_string())
    print('Evaluation finished!')


if __name__ == '__main__':
    app.run(main)
