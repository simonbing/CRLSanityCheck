import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cpu") # TODO: remove 0 if we want all cuda devices
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
