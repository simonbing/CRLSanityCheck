from typing import List

import numpy as np
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
