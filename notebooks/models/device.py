"""
A small util to get the right training device.
"""
import torch
import wandb

def get_available_device():
    device = wandb.config.device
    if device == "gpu":
        return torch.device("gpu" if torch.cuda.is_available() else "cpu")
    return device
