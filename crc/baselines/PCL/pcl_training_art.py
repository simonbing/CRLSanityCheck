""" Training
    Main script for training the model
"""


import os
import pickle
import shutil
import tarfile
from sklearn.decomposition import PCA

from subfunc.generate_artificial_data_art import generate_artificial_data
from pcl.train import train
from vae.vae_train import train as vae_train
from subfunc.showdata import *


# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_layer = 3  # number of layers of mixing-MLP
num_comp = 4  # number of components (dimension)
num_data = 2**20  # number of data points
ar_coef = [0.9] * num_comp  # AR(p) coefficients of components
ar_order = 1  # p of AR(p)
random_seed = 0  # random seed

# MLP ---------------------------------------------------------
list_hidden_nodes = [2 * num_comp] * (num_layer - 1) + [num_comp]
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]

model, initial_learning_rate = 'pcl', 0.1  # PCL
# model, initial_learning_rate = 'vae', 0.01  # VAE

# Training ----------------------------------------------------
momentum = 0.9  # momentum parameter of SGD
max_steps = int(2e5)  # number of iterations (mini-batches)
decay_steps = int(1e5)  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e7)  # interval to save checkpoint
summary_steps = int(1e4)  # interval to save summary
apply_pca = True  # apply PCA for preprocessing or not
weight_decay = 1e-5  # weight decay


# Other -------------------------------------------------------
# # Note: save folder must be under ./storage
train_dir_base = './storage'

train_dir = os.path.join(train_dir_base, 'model')

saveparmpath = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters


# =============================================================
# =============================================================

# Prepare save folder -----------------------------------------
if train_dir.find('/storage/') > -1:
    if os.path.exists(train_dir):
        print('delete savefolder: %s...' % train_dir)
        shutil.rmtree(train_dir)  # remove folder
    print('make savefolder: %s...' % train_dir)
    os.makedirs(train_dir)  # make folder
else:
    assert False, 'savefolder looks wrong'

# Generate sensor signal --------------------------------------
x, s, y, _, _, _ = generate_artificial_data(num_comp=num_comp,
                                            num_data=num_data,
                                            ar_coef=ar_coef,
                                            ar_order=ar_order,
                                            num_layer=num_layer,
                                            random_seed=random_seed)

# Preprocessing -----------------------------------------------
pca = PCA(whiten=True)
x = pca.fit_transform(x)

# Train model  ------------------------------------------------
if model == 'pcl':
    train(x,
          list_hidden_nodes=list_hidden_nodes,
          initial_learning_rate=initial_learning_rate,
          momentum=momentum,
          max_steps=max_steps,
          decay_steps=decay_steps,
          decay_factor=decay_factor,
          batch_size=batch_size,
          train_dir=train_dir,
          ar_order=ar_order,
          weight_decay=weight_decay,
          checkpoint_steps=checkpoint_steps,
          moving_average_decay=moving_average_decay,
          summary_steps=summary_steps,
          random_seed=random_seed)
elif model == 'vae':
    vae_train(x,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              max_steps=max_steps,
              decay_steps=decay_steps,
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              weight_decay=weight_decay,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              summary_steps=summary_steps,
              random_seed=random_seed)

# Save parameters necessary for evaluation --------------------
model_parm = {'random_seed': random_seed,
              'num_comp': num_comp,
              'num_data': num_data,
              'ar_coef': ar_coef,
              'ar_order': ar_order,
              'num_layer': num_layer,
              'model':model,
              'list_hidden_nodes': list_hidden_nodes,
              'moving_average_decay': moving_average_decay}

print('Save parameters...')
with open(saveparmpath, 'wb') as f:
    pickle.dump(model_parm, f, pickle.HIGHEST_PROTOCOL)

# Save as tarfile
tarname = train_dir + ".tar.gz"
archive = tarfile.open(tarname, mode="w:gz")
archive.add(train_dir, arcname="./")
archive.close()

print('done.')
