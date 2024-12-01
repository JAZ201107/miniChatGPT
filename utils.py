import numpy as np
import torch
import os
import random

from torch.utils.tensorboard import SummaryWriter


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    return device


def count_parameter(module):
    count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Number of parameters: {count}")
    return count


def seed_everything(seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)


def print_info(writer, **kwargs):
    step = kwargs["step"]
    s = ""
    for key, value in kwargs.items():
        if key == "dt":
            s += f"{key}: {value:.2f} ms | "
        else:
            s += f"{key}: {value} | "
        if key == "step" or key == "dt":
            continue
        writer.add_scalar(f"{key}", value, step)
    print(s)
    return None
