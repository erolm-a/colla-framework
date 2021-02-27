"""
A small util to get the right training device.
"""

import os
import torch

def get_available_device() -> torch.device:
    device = os.environ.get("COLLA_DEVICE", "cuda")
    
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
