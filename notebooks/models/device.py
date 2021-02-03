"""
A small util to get the right training device.
"""
import torch
import wandb

def get_available_device():
    device = wandb.config.device
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
