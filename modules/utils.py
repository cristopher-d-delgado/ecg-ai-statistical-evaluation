# modules/utils.py

"""
Shared utilities for ECG AI Statistical Validation project.
"""

import torch
import numpy as np
import random


def get_device() -> torch.device:
    """
    Automatically select the best available compute device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: The selected device.

    Example:
        >>> DEVICE = get_device()
        Using device: cuda
        GPU: NVIDIA GeForce RTX 3090
        VRAM: 24.0 GB
    """
    if torch.cuda.is_available():
        device = torch.device('cuda') # NVIDIA support
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Apple Support
    else:
        device = torch.device('cpu') # Neither use CPU

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds globally for reproducibility.
    Covers Python, NumPy, and PyTorch (CPU and GPU).

    Args:
        seed (int): Random seed. Default 42.

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)