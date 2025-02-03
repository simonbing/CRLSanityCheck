import torch


def get_device():
    """
    Get the torch available torch device.

    Returns
    -------
        device : torch.device
            Device for torch operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
