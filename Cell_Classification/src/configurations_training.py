import logging
from pathlib import Path

import numpy as np
import random
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Determine the device to use for computations.

    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logging.info(f"Device selected: {device}")
    logging.info(f"Number of GPUs available: {num_gpus}")
    return device, num_gpus


def create_directories(directories: list) -> None:
    """
    Create necessary directories if they don't exist.

    Args:
        directories (list): List of directory paths to create.
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory '{directory}' is ready.")




