from typing import List

import numpy as np
from sempler.lganm import _parse_interventions
import torch


def topk_gumbel_softmax(k, logits, tau, hard=True):
    """
    Applies the top-k Gumbel-Softmax operation to the input logits.

    Args:
        k (int): The number of elements to select from the logits.
        logits (torch.Tensor): The input logits.
        tau (float): The temperature parameter for the Gumbel-Softmax operation.
        hard (bool, optional): Whether to use the straight-through approximation.
            If True, the output will be a one-hot vector. If False, the output will be a
            continuous approximation of the top-k elements. Default is True.

    Returns:
        torch.Tensor: The output tensor after applying the top-k Gumbel-Softmax operation.
    """
    EPSILON = np.finfo(np.float32).tiny

    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    g = m.sample()
    logits = logits + g

    # continuous top k
    khot = torch.zeros_like(logits).type_as(logits)
    onehot_approx = torch.zeros_like(logits).type_as(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).type_as(logits))
        logits = logits + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(logits / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot
    return res


def gumbel_softmax_mask(avg_logits: torch.Tensor, subsets: List,
                        content_sizes: List):
    """
    Applies the Gumbel-Softmax function to generate masks for each subset.

    Args:
        avg_logits (torch.Tensor): The average logits for each subset.
        subsets (List): The list of subsets.
        content_sizes (List): The list of content sizes for each subset.

    Returns:
        List: The list of masks generated using Gumbel-Softmax for each subset.
    """
    masks = []
    for i, subset in enumerate(subsets):
        m = topk_gumbel_softmax(k=content_sizes[i], logits=avg_logits,
                                tau=1.0, hard=True)
        masks += [m]
    return masks


def sample_from_dag(W, lganm, n, do_interventions={}):
    """
    Taken from the Contrstive CRL code. Adaptation of the LGANM sempler sampling
    function for different noise distributions.

    Args:
        W (np.array): Adjacency matrix of the DAG.
        lganm (sempler.LGANM): Linear Gaussian additive noise model.
        n (int): Number of samples to draw.
        do_interventions (list): List of sempler do-interventions.

    Returns:
        None
    """
    variances = lganm.variances.copy()
    means = lganm.means.copy()
    if do_interventions:
        do_interventions = _parse_interventions(do_interventions)
        targets = do_interventions[:, 0].astype(int)
        means[targets] = do_interventions[:, 1]
        variances[targets] = do_interventions[:, 2]
        W[:, targets] = 0

    # Sampling by building the joint distribution
    A = np.linalg.inv(np.eye(lganm.p) - W.T)

    # Draw Gaussian noise
    noise_variables = self.sample_noise_variables(n)
    std_reshape = np.reshape(np.sqrt(variances), (1, self.d))
    mean_reshape = np.reshape(means, (1, self.d))
    noise_variables = std_reshape * noise_variables + mean_reshape
    return (A @ noise_variables.T).T
