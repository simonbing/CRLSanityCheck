import copy

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


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


def compute_multiview_r2(z, z_hat, content_indices, subsets):
    """
    Args:
        z (nd.array, (n_samples, dim)): Ground truth latents.
        z_hat (nd.array, (n_views, n_samples, dim)): Estimated latents

    Returns:
        None
    """
    # Loop over all "modalities", in our case this is each view/encoder
    # Get the relevant factors of each encoder (i.e. the content)-> actually, all latent dims are taken!
    # Loop over aforementioned factors
    # Loop over the subsets of views that share some content
    # Get encodings of these subsets as well as ground truth factors
    # Get linear and nonlinear r2 scores
    n_views = z_hat.shape[0]
    n_subsets = len(subsets)
    n_factors = z.shape[1]

    results_lin = np.empty((n_factors, n_subsets, n_views))
    results_lin[:] = np.nan

    results_nonlin = np.empty((n_factors, n_subsets, n_views))
    results_nonlin[:] = np.nan

    for i in range(n_factors):
        # TODO: think if we want to only take the content factors as targets (ie loop over subsets)
        target_factor = z[:, i]
        for j, subset in enumerate(subsets):
            for view_idx in subset:
                source_factors = z_hat[view_idx, :, content_indices[j]]
                # Split into train test
                source_train, source_test, target_train, target_test = \
                    train_test_split(source_factors.T, target_factor, test_size=0.2)
                # Standardize encodings
                scaler = StandardScaler()
                source_train = scaler.fit_transform(source_train)
                source_test = scaler.transform(source_test)

                # Linear
                lin_reg = LinearRegression(n_jobs=-1)
                lin_reg.fit(source_train, target_train)

                results_lin[i, j, view_idx] = r2_score(target_test,
                                                       lin_reg.predict(source_test))

                # Nonlinear
                nonlin_reg = MLPRegressor(max_iter=1000)
                nonlin_reg.fit(source_train, target_train)

                results_nonlin[i, j, view_idx] = r2_score(target_test,
                                                          nonlin_reg.predict(source_test))

    # Average results over views within one subset
    avg_results_lin = np.nanmean(results_lin, axis=-1)
    avg_results_nonlin = np.nanmean(results_nonlin, axis=-1)

    return {'r2_lin': results_lin,
            'r2_nonlin': results_nonlin,
            'avg_r2_lin': avg_results_lin,
            'avg_r2_nonlin': avg_results_nonlin}
