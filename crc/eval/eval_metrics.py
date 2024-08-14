import copy

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def compute_MCC(z_hat, z, batch_size=5000):
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


def mean_corr_coef_np(z, z_hat, method='pearson', indices=None):
    """
    Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py

    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    x, y = z, z_hat
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]  # z x z_hat
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))

    cc = np.abs(cc)
    if indices is not None:
        cc_program = cc[:, indices[-d:]]
    else:
        cc_program = cc

    assignments = linear_sum_assignment(-1 * cc_program)
    score = cc_program[assignments].mean()

    perm_mat = np.zeros((d, d))
    perm_mat[assignments] = 1
    # cc_program_perm = np.matmul(perm_mat.transpose(), cc_program)
    cc_program_perm = np.matmul(cc_program,
                                perm_mat.transpose())  # permute the learned latents

    return score, cc_program_perm, assignments


def compute_SHD(G, G_hat, thresh=0.3):
    """
    Compute multiple versions of the SHD metric.

    Source code from code accompanying paper "Learning Linear Causal
    Representations from Interventions under General Nonlinear Mixing".

    Args:
        G, G_hat: (np.array [d, d]) Ground truth and learned adjacency matrix.
        thresh: (float) Edge strength threshold to keep an edge.
    """
    # TODO: don't we have to get the correct permutation from linear sum
    #  assignment? Maybe not if G_hat is automatically upper triangular...
    # Compute SHD with fixed threshold
    B = np.where(np.abs(G) > 0.01, 1, 0)
    np.fill_diagonal(G_hat, 0)
    B_hat = np.where(np.abs(G_hat) > thresh, 1, 0)
    np.fill_diagonal(B_hat, 0)
    shd = np.sum(np.abs(B - B_hat))

    # Compute SHD with optimal threshold
    thresholds = np.arange(0, 2, 0.005)
    min_shd = np.inf
    for t in thresholds:
        B_hat = np.where(np.abs(G_hat) > t, 1, 0)
        np.fill_diagonal(B_hat, 0)
        shd_t = np.sum(np.abs(B - B_hat))
        if shd_t < min_shd:
            min_shd = shd
    shd_opt = min_shd

    return shd, shd_opt
