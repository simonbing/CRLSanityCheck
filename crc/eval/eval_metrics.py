import copy

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_MCC(z_hat, z, batch_size=100):
    """
    Args:
        z_hat: np.array[n_samples, n_components]
            Learned sources.
        z: np.array[n_samples, n_gt_components]
            Ground truth sources.
        batch_size: int
            Batch size.

    Returns:
        mcc_arr: np.array[n_samples, n_components]
            MCC scores.
    """
    num_samples = z_hat.shape[0]
    latent_dim = z_hat.shape[1]
    total_batches = int(num_samples/batch_size)

    mcc_arr = []
    for batch_idx in range(total_batches):

        z_hat_batch = copy.deepcopy(z_hat[batch_idx*batch_size:(batch_idx+1)*batch_size])
        z_batch = copy.deepcopy(z[batch_idx*batch_size:(batch_idx+1)*batch_size])
        batch_idx += 1

        cross_corr = np.zeros((latent_dim, latent_dim))
        for i in range(latent_dim):
            for j in range(latent_dim):
                cross_corr[i,j] = (np.cov(z_hat_batch[:,i], z_batch[:,j])[0,1])/(np.std(z_hat_batch[:,i])*np.std(z_batch[:,j]))

        cost = -1*np.abs(cross_corr)
        row_ind, col_ind = linear_sum_assignment(cost)
        score = 100*(-1*cost[row_ind, col_ind].sum())/latent_dim
        # print(-100*cost[row_ind, col_ind])

        mcc_arr.append(score)

    return np.mean(mcc_arr)
