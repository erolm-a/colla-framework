"""
Provide top-level device configurations.
"""
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
