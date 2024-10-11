"""PCL training"""


from datetime import datetime
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb

from crc.baselines.PCL.pcl import pcl
from crc.baselines.PCL.subfunc.showdata import *


# =============================================================
# =============================================================
def train(data,
          epochs,
          list_hidden_nodes,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
          in_dim,
          latent_dim=4,
          ar_order=1,
          weight_decay=0,
          moving_average_decay=0.999,
          summary_steps=500,
          checkpoint_steps=10000,
          save_file='model.pt',
          load_file=None,
          random_seed=None):
    """Build and train a model
    Args:
        data: data. 2D ndarray [num_data, num_comp]
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        initial_learning_rate: initial learning rate
        momentum: momentum parameter (tf.train.MomentumOptimizer)
        max_steps: number of iterations (mini-batches)
        decay_steps: decay steps (tf.train.exponential_decay)
        decay_factor: decay factor (tf.train.exponential_decay)
        batch_size: mini-batch size
        train_dir: save directory
        ar_order: model order of AR
        weight_decay: weight decay
        moving_average_decay: (option) moving average decay of variables to be saved
        summary_steps: (option) interval to save summary
        checkpoint_steps: (option) interval to save checkpoint
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        random_seed: (option) random seed
    Returns:
    """

    # set random_seed
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # num_data, in_dim = data.shape

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # define network
    model = pcl.Net(h_sizes=list_hidden_nodes,
                    in_dim=in_dim,
                    latent_dim=latent_dim,
                    ar_order=ar_order)
    model = model.to(device)
    model.train()

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
    if type(decay_steps) == list:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=decay_factor)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)
    writer = SummaryWriter(log_dir=train_dir)

    state_dict_ema = model.state_dict()

    trained_step = 0
    if load_file is not None:
        print('Load trainable parameters from %s...' % load_file)
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trained_step = checkpoint['step']

    for i in range(epochs):
        # training iteration
        for batch_data in data:  # iterate over dataloader
        # for step in range(trained_step, max_steps):
            start_time = time.time()

            # make shuffled batch
            # t_idx = np.random.choice(num_data - ar_order, batch_size) + ar_order
            # t_idx_ar = t_idx.reshape([-1, 1]) + np.arange(0, -ar_order - 1, -1).reshape([1, -1])
            # x_batch = data[t_idx_ar.reshape(-1), :].reshape([batch_size, ar_order + 1, -1])
            #
            # xast_batch = x_batch.copy()
            # for i in range(ar_order):
            #     tast_idx = np.random.choice(num_data, batch_size)
            #     xast_batch[:, i + 1, :] = data[tast_idx, :]
            #
            # x_torch = torch.from_numpy(np.concatenate([x_batch, xast_batch], axis=0).astype(np.float32)).to(device)
            # y_torch = torch.cat([torch.ones([batch_size]), torch.zeros([batch_size])]).to(device)

            x = batch_data[0]
            x_perm = batch_data[1]
            y = batch_data[2]
            y_perm = batch_data[3]

            x_torch = torch.cat((x, x_perm))
            y_torch = torch.squeeze(torch.cat((y, y_perm)))

            x_torch = x_torch.to(device)
            y_torch = y_torch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, h = model(x_torch)
            loss = criterion(logits, y_torch)
            loss.backward()
            optimizer.step()

            model.a.data = model.a.data.clamp(min=0)

            # moving average of parameters
            state_dict_n = model.state_dict()
            for key in state_dict_ema:
                state_dict_ema[key] = moving_average_decay * state_dict_ema[key] \
                                      + (1.0 - moving_average_decay) * state_dict_n[key]

            # accuracy
            predicted = (logits > 0.5).float()
            accu_val = (predicted == y_torch).sum().item()/(batch_size*2)
            loss_val = loss.item()
            lr = scheduler.get_last_lr()[0]
            # wandb logging
            wandb.log({'loss': loss.item(),
                       'accuracy': accu_val})

            duration = time.time() - start_time

            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

            # display stats
            # if step % 100 == 0:
            #     num_examples_per_step = batch_size
            #     examples_per_sec = num_examples_per_step / duration
            #     sec_per_batch = float(duration)
            #     format_str = '%s: step %d, lr = %f, loss = %.2f, accuracy = %3.2f (%.1f examples/sec; %.3f sec/batch)'
            #     print(format_str % (datetime.now(), step, lr, loss_val, accu_val * 100,
            #                         examples_per_sec, sec_per_batch))

        scheduler.step()

        # save summary
        # if i % summary_steps == 0:
        #     writer.add_scalar('scalar/lr', lr, i)
        #     writer.add_scalar('scalar/loss', loss_val, i)
        #     writer.add_scalar('scalar/accu', accu_val, i)
        #     h_val = h.cpu().detach().numpy()
        #     h_comp = np.split(h_val, indices_or_sections=h.shape[1], axis=1)
        #     for (i, cm) in enumerate(h_comp):
        #         writer.add_histogram('h/h%d' % i, cm)
        #     for k, v in state_dict_n.items():
        #         writer.add_histogram('w/%s' % k, v)

        # save the model checkpoint periodically.
        if i % checkpoint_steps == 0:
            checkpoint_path = os.path.join(train_dir, save_file)
            torch.save({'step': i,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': state_dict_ema,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, checkpoint_path)

    # save trained model ----------------------------------
    save_path = os.path.join(train_dir, save_file)
    print('Save model in file: %s' % save_path)
    torch.save({'step': max_steps,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': state_dict_ema,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, save_path)
    torch.save(model, os.path.join(train_dir, 'best_model.pt'))
