from typing import List

import numpy as np
from sempler.lganm import _parse_interventions
import torch
from torch import nn


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


def construct_invertible_mlp(
    n: int = 20,
    n_layers: int = 2,
    n_iter_cond_thresh: int = 10000,
    cond_thresh_ratio: float = 0.25,
    weight_matrix_init = "pcl",
    act_fct = "leaky_relu",
):
    """
    Create an (approximately) invertible mixing network based on an MLP.
    Based on the mixing code by Hyvarinen et al.

    Args:
        n: Dimensionality of the input and output data
        n_layers: Number of layers in the MLP.
        n_iter_cond_thresh: How many random matrices to use as a pool to find weights.
        cond_thresh_ratio: Relative threshold how much the invertibility
            (based on the condition number) can be violated in each layer.
        weight_matrix_init: How to initialize the weight matrices.
        act_fct: Activation function for hidden layers.
    """

    class SmoothLeakyReLU(nn.Module):
        def __init__(self, alpha=0.2):
            super().__init__()
            self.alpha = alpha

        def forward(self, x):
            return self.alpha * x + (1 - self.alpha) * torch.log(1 + torch.exp(x))

    def get_act_fct(act_fct):
        if act_fct == "relu":
            return torch.nn.ReLU, {}, 1
        if act_fct == "leaky_relu":
            return torch.nn.LeakyReLU, {"negative_slope": 0.2}, 1
        elif act_fct == "elu":
            return torch.nn.ELU, {"alpha": 1.0}, 1
        elif act_fct == "max_out":
            raise NotImplementedError
        elif act_fct == "smooth_leaky_relu":
            return SmoothLeakyReLU, {"alpha": 0.2}, 1
        elif act_fct == "softplus":
            return torch.nn.Softplus, {"beta": 1}, 1
        else:
            raise Exception(f"activation function {act_fct} not defined.")

    layers = []
    act_fct, act_kwargs, act_fac = get_act_fct(act_fct)

    # Subfuction to normalize mixing matrix
    def l2_normalize(Amat, axis=0):
        # axis: 0=column-normalization, 1=row-normalization
        l2norm = np.sqrt(np.sum(Amat * Amat, axis))
        Amat = Amat / l2norm
        return Amat

    condList = np.zeros([n_iter_cond_thresh])
    if weight_matrix_init == "pcl":
        for i in range(n_iter_cond_thresh):
            A = np.random.uniform(-1, 1, [n, n])
            A = l2_normalize(A, axis=0)
            condList[i] = np.linalg.cond(A)
        condList.sort()  # Ascending order
    condThresh = condList[int(n_iter_cond_thresh * cond_thresh_ratio)]
    # print("condition number threshold: {0:f}".format(condThresh))

    for i in range(n_layers):
        lin_layer = nn.Linear(n, n, bias=False)

        if weight_matrix_init == "pcl":
            condA = condThresh + 1
            while condA > condThresh:
                weight_matrix = np.random.uniform(-1, 1, (n, n))
                weight_matrix = l2_normalize(weight_matrix, axis=0)

                condA = np.linalg.cond(weight_matrix)
                # print("    L{0:d}: cond={1:f}".format(i, condA))
            # print(f"layer {i+1}/{n_layers},  condition number: {np.linalg.cond(weight_matrix)}")
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)

        elif weight_matrix_init == "rvs":
            weight_matrix = ortho_group.rvs(n)
            lin_layer.weight.data = torch.tensor(weight_matrix, dtype=torch.float32)
        elif weight_matrix_init == "expand":
            pass
        else:
            raise Exception(f"weight matrix {weight_matrix_init} not implemented")

        layers.append(lin_layer)

        if i < n_layers - 1:
            layers.append(act_fct(**act_kwargs))

    mixing_net = nn.Sequential(*layers)

    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False

    return mixing_net
