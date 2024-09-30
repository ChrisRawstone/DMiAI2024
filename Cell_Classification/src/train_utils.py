import os
import cv2
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models
from torchvision.models import (
    ViT_B_32_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    MobileNet_V3_Large_Weights,
    Swin_V2_B_Weights,  # Added Swin V2 B Weights
)
from data.make_dataset import LoadTifDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Optuna for hyperparameter tuning
import optuna


def save_sample_images(dataset, num_samples=5, folder="plots", split="train"):
    """
    Saves a few sample images from the dataset to verify correctness.

    Args:
        dataset (CustomDataset): The dataset to sample from.
        num_samples (int, optional): Number of samples to save. Defaults to 5.
        folder (str, optional): Directory to save images. Defaults to 'plots'.
        split (str, optional): Dataset split name ('train' or 'val'). Defaults to 'train'.
    """
    os.makedirs(folder, exist_ok=True)
    for i in range(num_samples):
        image, label = dataset[i]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (
            image_np * np.array([0.229, 0.229, 0.229])
        ) + np.array([0.485, 0.485, 0.485])  # Unnormalize
        image_np = np.clip(image_np, 0, 1)
        plt.imshow(image_np)
        plt.title(f"{split.capitalize()} Label: {label}")
        plt.axis("off")
        plt.savefig(f"{folder}/{split}_sample_{i}.png")
        plt.close()

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        """
        Initializes the Focal Loss function.

        Args:
            alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter for modulating factor (1-p). Defaults to 2.
            logits (bool, optional): Whether inputs are logits. Defaults to True.
            reduce (bool, optional): Whether to reduce the loss. Defaults to True.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.

        Args:
            inputs (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        targets = targets.type_as(inputs)
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        else:
            BCE_loss = nn.functional.binary_cross_entropy(
                inputs, targets, reduction="none"
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        

