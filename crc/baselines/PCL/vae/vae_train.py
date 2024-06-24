"""VAE training"""


from datetime import datetime
import os.path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vae import vae
from subfunc.showdata import *


# =============================================================
# =============================================================
def train(data,
          list_hidden_nodes,
          initial_learning_rate,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
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

    num_data, num_dim = data.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define network
    model = vae.VariationalAutoencoder(h_sizes=list_hidden_nodes,
                                       num_dim=num_dim)
    model = model.to(device)
    model.train()

    # define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
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

    # training iteration
    for step in range(trained_step, max_steps):
        start_time = time.time()

        # make shuffled batch
        t_idx = np.random.choice(num_data, batch_size)
        x_batch = data[t_idx, :]

        x_torch = torch.from_numpy(x_batch.astype(np.float32)).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x_hat = model(x_torch)
        loss = ((x_torch - x_hat) ** 2).sum() + model.encoder.kl
        loss.backward()
        optimizer.step()
        scheduler.step()

        # moving average of parameters
        state_dict_n = model.state_dict()
        for key in state_dict_ema:
            state_dict_ema[key] = moving_average_decay * state_dict_ema[key] \
                                  + (1.0 - moving_average_decay) * state_dict_n[key]

        # accuracy
        loss_val = loss.item()
        lr = scheduler.get_last_lr()[0]

        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # display stats
        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: step %d, lr = %f, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), step, lr, loss_val,
                                examples_per_sec, sec_per_batch))

        # save summary
        if step % summary_steps == 0:
            writer.add_scalar('scalar/lr', lr, step)
            writer.add_scalar('scalar/loss', loss_val, step)
            # writer.add_scalar('scalar/accu', accu_val, step)
            h_val = x_hat.cpu().detach().numpy()
            h_comp = np.split(h_val, indices_or_sections=x_hat.shape[1], axis=1)
            for (i, cm) in enumerate(h_comp):
                writer.add_histogram('h/h%d' % i, cm)
            for k, v in state_dict_n.items():
                writer.add_histogram('w/%s' % k, v)

        # save the model checkpoint periodically.
        if step % checkpoint_steps == 0:
            checkpoint_path = os.path.join(train_dir, save_file)
            torch.save({'step': step,
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
