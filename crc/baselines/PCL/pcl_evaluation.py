""" Evaluation
    Main script for evaluating the model trained by pcl_training.py
"""


import os
import numpy as np
import pickle
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from subfunc.generate_artificial_data import generate_artificial_data
from subfunc.showdata import *
from pcl import pcl, utils


# parameters ==================================================
# =============================================================

eval_dir_base = './storage'

eval_dir = os.path.join(eval_dir_base, 'model')

parmpath = os.path.join(eval_dir, 'parm.pkl')
savefile = eval_dir.replace('.tar.gz', '') + '.pkl'

load_ema = True  # recommended unless the number of iterations was not enough

num_data_test = -1  # number of data points for testing (-1: same with training)
# num_data_test = None  # do not generate test data


# =============================================================
# =============================================================

if eval_dir.find('.tar.gz') >= 0:
    unzipfolder = './storage/temp_unzip'
    utils.unzip(eval_dir, unzipfolder)
    eval_dir = unzipfolder
    parmpath = os.path.join(unzipfolder, 'parm.pkl')

modelpath = os.path.join(eval_dir, 'model.pt')

# Load parameter file
with open(parmpath, 'rb') as f:
    model_parm = pickle.load(f)

num_comp = model_parm['num_comp']
num_data = model_parm['num_data']
ar_coef = model_parm['ar_coef']
ar_order = model_parm['ar_order']
num_layer = model_parm['num_layer']
list_hidden_nodes = model_parm['list_hidden_nodes']
moving_average_decay = model_parm['moving_average_decay']
random_seed = model_parm['random_seed']
if num_data_test == -1:
    num_data_test = num_data

# Generate sensor signal --------------------------------------
x, s, y, x_te, s_te, y_te = generate_artificial_data(num_comp=num_comp,
                                                     num_data=num_data,
                                                     num_data_test=num_data_test,
                                                     ar_coef=ar_coef,
                                                     ar_order=ar_order,
                                                     num_layer=num_layer,
                                                     random_seed=random_seed)

# Preprocessing -----------------------------------------------
pca = PCA(whiten=True)
x = pca.fit_transform(x)
if x_te is not None:
    x_te = pca.transform(x_te)

# Evaluate model ----------------------------------------------
# -------------------------------------------------------------

# define network
model = pcl.Net(h_sizes=list_hidden_nodes,
                num_dim=num_comp)
device = 'cpu'
model = model.to(device)
model.eval()

# load parameters
print('Load trainable parameters from %s...' % modelpath)
checkpoint = torch.load(modelpath, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
if load_ema:
    model.load_state_dict(checkpoint['ema_state_dict'])

# augment data for AR model
t_idx = np.arange(x.shape[0] - ar_order) + ar_order
t_idx = t_idx.reshape([-1, 1]) + np.arange(0, -ar_order - 1, -1).reshape([1, -1])
xt = x[t_idx.reshape(-1), :].reshape([-1, ar_order + 1, x.shape[-1]])
xast = xt.copy()
for i in range(ar_order):
    xast[:, i + 1, :] = x[np.random.choice(num_data, xt.shape[0]), :]

# forward
x_torch = torch.from_numpy(np.concatenate([xt, xast], axis=0).astype(np.float32)).to(device)
y_torch = torch.cat([torch.ones([xt.shape[0]]), torch.zeros([xast.shape[0]])]).to(device)

logits, h = model(x_torch)
pred = (logits > 0).float()
h, hast = torch.split(h, split_size_or_sections=int(h.size()[0]/2), dim=0)

# convert to numpy
pred_val = pred.cpu().numpy()
y_val = y_torch.cpu().numpy()
h_val = np.squeeze(h[:, 0, :].detach().cpu().numpy())

# for test data
if x_te is not None:
    t_idx = np.arange(x_te.shape[0] - ar_order) + ar_order
    t_idx = t_idx.reshape([-1, 1]) + np.arange(0, -ar_order - 1, -1).reshape([1, -1])
    xt_te = x_te[t_idx.reshape(-1), :].reshape([-1, ar_order + 1, x.shape[-1]])
    xast_te = xt_te.copy()
    for i in range(ar_order):
        xast_te[:, i + 1, :] = x_te[np.random.choice(num_data, xt.shape[0]), :]

    x_te_torch = torch.from_numpy(np.concatenate([xt_te, xast_te], axis=0).astype(np.float32)).to(device)
    y_te_torch = torch.cat([torch.ones([xt_te.shape[0]]), torch.zeros([xast_te.shape[0]])]).to(device)
    logits_te, h_te = model(x_te_torch)
    pred_te = (logits_te > 0).float()
    pred_te_val = pred_te.cpu().numpy()
    y_te_val = y_te_torch.cpu().numpy()


# Evaluate outputs --------------------------------------------
# -------------------------------------------------------------

# Calculate accuracy
accu_tr = accuracy_score(pred_val, y_val.T)
if x_te is not None:
    accu_te = accuracy_score(pred_te_val, y_te_val)

# correlation
corrmat_tr, sort_idx, _ = utils.correlation(h_val, s[1:, :], 'Pearson')

meanabscorr_tr = np.mean(np.abs(np.diag(corrmat_tr)))

showmat(corrmat_tr,
        yticklabel=np.arange(num_comp),
        xticklabel=sort_idx,
        ylabel='source',
        xlabel='feature')

# Display results
print('Result...')
print('    accuracy (train) : %7.4f [percent]' % (accu_tr * 100))
if 'accu_te' in locals():
    print('    accuracy (test)  : %7.4f [percent]' % (accu_te * 100))
print('    correlation      : %7.4f' % meanabscorr_tr)

# Save results
result = {'accu_tr': accu_tr,
          'accu_te': accu_te if 'accu_te' in locals() else None,
          'corrmat_tr': corrmat_tr,
          'meanabscorr_tr': meanabscorr_tr,
          'sort_idx': sort_idx,
          'num_comp': num_comp,
          'modelpath': modelpath}

print('Save results...')
with open(savefile, 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

print('done.')
