import torch


def infonce_base_loss(hz_subset, content_indices, sim_metric, criterion, projector=None, tau=1.0):
    """
    Computes the InfoNCE (Normalized Cross Entropy) loss for multi-view data.

    Args:
        hz_subset (list): List of tensors representing the latent space of each view.
        content_indices (list): List of indices representing the content dimensions.
        sim_metric (function): Similarity metric function to compute pairwise similarities.
        criterion (function): Loss criterion function.
        projector (function, optional): Projection function to project the latent space. Defaults to None.
        tau (float, optional): Temperature parameter for similarity computation. Defaults to 1.0.

    Returns:
        torch.Tensor: Total loss value.

    """

    n_view = len(hz_subset)
    SIM = [
        [torch.Tensor().type_as(hz_subset) for _ in range(n_view)] for _ in range(n_view)
    ]  # n_views x n_view x batch_size (d) x batch_size (d)

    projector = projector or (lambda x: x)

    for i in range(n_view):
        for j in range(n_view):
            if j >= i:
                # compute similarity matrix using projected latents
                sim_ij = (
                    sim_metric(  # (hz[i]: n_views, n_latent_dim)
                        projector(hz_subset[i].unsqueeze(-2)),
                        projector(
                            hz_subset[j].unsqueeze(-3)
                        ),  # (bs, 1, n_latent_dim) and (1, bs n_latent_dim) -> bs , bs
                    )
                    / tau
                ).type_as(hz_subset)
                # compute positive pairs using (diagonal elements) only the content dimensions
                pos_sim_ij = (
                    sim_metric(  # (hz[i]: n_views, n_latent_dim)
                        hz_subset[i].unsqueeze(-2)[..., content_indices],
                        hz_subset[j].unsqueeze(-3)[
                            ..., content_indices
                        ],  # (bs, 1, n_latent_dim) and (1, bs n_latent_dim) -> bs , bs
                    )
                    / tau
                ).type_as(hz_subset)
                sim_ij = pos_sim_ij
                if i == j:
                    d = sim_ij.shape[-1]  # batch size
                    sim_ij[..., range(d), range(d)] = float("-inf")
                SIM[i][j] = sim_ij
            else:
                SIM[i][j] = SIM[j][i].transpose(-1, -2).type_as(hz_subset)

    total_loss_value = torch.zeros(1).type_as(hz_subset)
    for i in range(n_view):
        for j in range(n_view):
            if i < j:
                raw_scores = []
                raw_scores1 = torch.cat([SIM[i][j], SIM[i][i]], dim=-1).type_as(hz_subset)
                raw_scores2 = torch.cat([SIM[j][j], SIM[j][i]], dim=-1).type_as(hz_subset)
                raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)  # d, 2d
                targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
                total_loss_value += criterion(raw_scores, targets)
    return total_loss_value


def infonce_loss(hz, sim_metric, criterion, projector=None, tau=1.0, estimated_content_indices=None, subsets=None):
    """
    Calculates the sum of InfoNCE loss for a given input tensor `hz`, over all subsets.

    Args:
        hz (torch.Tensor): The input tensor of shape (batch_size, ..., num_features).
        sim_metric: The similarity metric used for calculating the loss.
        criterion: The loss criterion used for calculating the loss.
        projector: The projector used for projecting the input tensor (optional).
        tau (float): The temperature parameter for the loss calculation (default: 1.0).
        estimated_content_indices: The estimated content indices (optional).
        subsets: The subsets of indices used for calculating the loss (optional).

    Returns:
        torch.Tensor: The calculated InfoNCE loss.

    """
    if estimated_content_indices is None:
        return infonce_base_loss(hz, sim_metric, criterion, projector, tau)
    else:
        total_loss = torch.zeros(1).type_as(hz)
        for est_content_indices, subset in zip(estimated_content_indices, subsets):
            total_loss += infonce_base_loss(
                hz[list(subset), ...], est_content_indices, sim_metric, criterion, projector, tau
            )
        return total_loss