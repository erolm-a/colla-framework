"""
Provide top-level device configurations.
"""
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def global_move_to_cpu():
    """
    Move to CPU
    """
    global DEVICE
    DEVICE = "cpu"