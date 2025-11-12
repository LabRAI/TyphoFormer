#utils.py
import torch


def print_shape(name, tensor, indent=0):
    """Print Tensor's name and Dimension, used for debug"""
    prefix = " " * indent
    if tensor is None:
        print(f"{prefix}{name}: None")
    elif isinstance(tensor, (list, tuple)):
        print(f"{prefix}{name}: {[t.shape for t in tensor]}")
    elif isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}: {tuple(tensor.shape)}")
    else:
        print(f"{prefix}{name}: {type(tensor)}")
