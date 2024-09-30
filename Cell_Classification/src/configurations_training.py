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


def setup_logging(log_dir: Path, log_file: str = "training.log") -> None:
    """
    Configure logging to file and console.

    Args:
        log_dir (Path): Directory where the log file will be saved.
        log_file (str, optional): Log file name. Defaults to "training.log".
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")


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




