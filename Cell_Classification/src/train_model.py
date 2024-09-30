# src/train_model.py

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

from src.utils import calculate_custom_score

# ===============================
# 1. Setup and Configuration
# ===============================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define device and check available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Setup logging early to capture GPU info
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info(f"Number of GPUs available: {num_gpus}")

# Create directories
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===============================
# 2. Custom Score Function
# ===============================

# Assuming calculate_custom_score is defined in src/utils.py

# ===============================
# 4. Get Data and Dataloaders
# ===============================

def get_dataloaders(
    batch_size, img_size, fold_indices=None, num_workers=8, pin_memory=True
):
    """
    Returns DataLoader objects for training, cross-validation, and original validation datasets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Image size for resizing.
        fold_indices (tuple, optional): Tuple containing train and val indices for cross-validation.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 8.
        pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory. Defaults to True.

    Returns:
        tuple: Depending on fold_indices, returns training and validation DataLoaders.
    """
    # Define transformations
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=(0.485, 0.485, 0.485),
                std=(0.229, 0.229, 0.229),
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.485, 0.485),
                std=(0.229, 0.229, 0.229),
            ),
            ToTensorV2(),
        ]
    )

    # Paths to images and CSV files
    image_dir = "data/training"
    csv_file_path = "data/training.csv"

    image_dir_original_val = "data/validation16bit"
    csv_file_path_original_val = "data/validation.csv"

    # Create the datasets
    full_train_dataset = LoadTifDataset(
        image_dir=image_dir, csv_file_path=csv_file_path, transform=train_transform
    )
    original_val_dataset = LoadTifDataset(
        image_dir=image_dir_original_val,
        csv_file_path=csv_file_path_original_val,
        transform=val_transform,
    )

    if fold_indices is not None:
        train_indices, val_indices = fold_indices
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # If no fold_indices provided, use the entire training set
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = None

    # Original validation loader
    original_val_loader = DataLoader(
        original_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, original_val_loader

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

# Uncomment the following lines if you want to save sample images
# train_loader_temp, val_loader_temp, original_val_loader_temp = get_dataloaders(batch_size=1, img_size=224)
# save_sample_images(train_loader_temp.dataset, split='train')
# save_sample_images(val_loader_temp.dataset, split='val')
# save_sample_images(original_val_loader_temp.dataset, split='original_val')

# logging.info("Sample images saved to 'plots' folder.")

# ===============================
# 9. Define Models Dictionary
# ===============================

def get_models(model_name, num_classes=1):
    """
    Returns a specific model based on the model_name.

    Args:
        model_name (str): Name of the model to retrieve.
        num_classes (int, optional): Number of output classes. Defaults to 1.

    Returns:
        nn.Module: The specified model.
    """
    if model_name == "ViT16":
        vit_weights = ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=vit_weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == "ViT32":
        vit_weights32 = ViT_B_32_Weights.DEFAULT
        model = models.vit_b_32(weights=vit_weights32)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == "ResNet18":
        resnet18_weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=resnet18_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "ResNet50":
        resnet50_weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=resnet50_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "ResNet101":
        resnet101_weights = ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=resnet101_weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "EfficientNetB0":
        efficientnet_b0_weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=efficientnet_b0_weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "EfficientNetB4":
        efficientnet_b4_weights = EfficientNet_B4_Weights.DEFAULT
        model = models.efficientnet_b4(weights=efficientnet_b4_weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "MobileNetV3":
        mobilenet_v3_weights = MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=mobilenet_v3_weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "DenseNet121":
        densenet_weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=densenet_weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model

# ===============================
# 10. Define Focal Loss with Class Weights
# ===============================

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

# ===============================
# 11. Hyperparameter Tuning with Optuna
# ===============================

def get_model_parallel(model):
    """
    Wraps the model with DataParallel if multiple GPUs are available.

    Args:
        model (nn.Module): The model to wrap.

    Returns:
        nn.Module: The potentially parallelized model.
    """
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        logging.info("Using a single GPU.")
    return model

# Initialize a global list to keep top 3 models
top_k = 3
top_models = []

def update_top_models(model_state_dict, custom_score, trial_number, fold_number):
    """
    Updates the global top_models list with the new model if it is among the top_k.

    Args:
        model_state_dict (dict): The state dictionary of the model.
        custom_score (float): The custom score of the model.
        trial_number (int): The trial number from Optuna.
        fold_number (int): The fold number in cross-validation.
    """
    global top_models
    if len(top_models) < top_k:
        top_models.append(
            {
                "score": custom_score,
                "state_dict": model_state_dict,
                "trial": trial_number,
                "fold": fold_number,
            }
        )
        top_models = sorted(top_models, key=lambda x: x["score"], reverse=True)
    else:
        if custom_score > top_models[-1]["score"]:
            top_models[-1] = {
                "score": custom_score,
                "state_dict": model_state_dict,
                "trial": trial_number,
                "fold": fold_number,
            }
            top_models = sorted(top_models, key=lambda x: x["score"], reverse=True)

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization with 5-fold cross-validation.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Average validation custom score across all folds.
    """
    # Hyperparameters to tune
    model_name = trial.suggest_categorical(
        "model_name",
        [
            "ViT16",
            "ViT32",
            "ResNet18",
            "ResNet50",
            "ResNet101",
            "EfficientNetB0",
            "EfficientNetB4",
            "MobileNetV3",
            "DenseNet121",
        ],
    )
    if (model_name == 'ViT16' or model_name == 'ViT32'):
        img_size = 224  # Fixed for ViT
    else:
        img_size = trial.suggest_categorical("img_size", [128, 224, 256, 299, 331, 350, 400, 500])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)

    num_epochs = 2  # For quick experimentation; increase for better results
    n_splits = 3  # 5-fold cross-validation

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Load full training dataset
    full_train_dataset = LoadTifDataset(
        image_dir="data/training8bit",
        csv_file_path="data/training.csv",
        transform=A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Normalize(
                    mean=(0.485, 0.485, 0.485),
                    std=(0.229, 0.229, 0.229),
                ),
                ToTensorV2(),
            ]
        ),
    )

    # Get labels for StratifiedKFold
    labels = full_train_dataset.labels_df["is_homogenous"].values

    # Initialize list to store scores for each fold
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logging.info(f"Starting Fold {fold + 1}/{n_splits}")

        # Create DataLoaders for this fold
        train_loader, val_loader, original_val_loader = get_dataloaders(
            batch_size=batch_size,
            img_size=img_size,
            fold_indices=(train_idx, val_idx),
        )

        # Initialize model
        model = get_models(model_name=model_name)
        model = get_model_parallel(model)
        model = model.to(device)

        # Define loss, optimizer, scheduler
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler()

        best_fold_score = 0

        for epoch in range(num_epochs):
            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}")

            # Training Phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            progress_bar = tqdm(train_loader, desc=f"Fold {fold +1} Training Epoch {epoch+1}", leave=False)
            for images, labels_batch in progress_bar:
                images = images.to(device, non_blocking=True)
                labels_batch = labels_batch.float().unsqueeze(1).to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(labels_batch.detach().cpu().numpy())

                progress_bar.set_postfix({"Loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation Phase (New Validation Set)
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Fold {fold +1} Validation", leave=False)
                for images, labels_batch in progress_bar:
                    images = images.to(device, non_blocking=True)
                    labels_batch = labels_batch.float().unsqueeze(1).to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)

                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels_batch.detach().cpu().numpy())

            avg_val_loss = val_loss / len(val_loader.dataset)

            # Original Validation Set Evaluation
            original_val_loss = 0
            original_val_preds = []
            original_val_targets = []

            with torch.no_grad():
                progress_bar = tqdm(original_val_loader, desc=f"Fold {fold +1} Original Validation", leave=False)
                for images, labels_batch in progress_bar:
                    images = images.to(device, non_blocking=True)
                    labels_batch = labels_batch.float().unsqueeze(1).to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)

                    original_val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                    original_val_preds.extend(preds)
                    original_val_targets.extend(labels_batch.detach().cpu().numpy())

            avg_original_val_loss = original_val_loss / len(original_val_loader.dataset)

            # Calculate custom scores
            preds_binary = (np.array(val_preds) > 0.5).astype(int)
            custom_score = calculate_custom_score(val_targets, preds_binary)

            preds_binary_original = (np.array(original_val_preds) > 0.5).astype(int)
            custom_score_original = calculate_custom_score(original_val_targets, preds_binary_original)

            # Log the scores
            logging.info(
                f"Fold {fold +1} Epoch {epoch +1}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Original Val Loss: {avg_original_val_loss:.4f} | "
                f"Custom Score (New Val): {custom_score:.4f} | "
                f"Custom Score (Original Val): {custom_score_original:.4f}"
            )

            # Update scheduler
            scheduler.step()

            # Update best score and save model if necessary
            if custom_score > best_fold_score:
                best_fold_score = custom_score

                # Save the best model for this fold and trial
                save_path = f"checkpoints/best_model_trial_{trial.number}_fold_{fold +1}.pth"
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                logging.info(
                    f"Fold {fold +1} Epoch {epoch +1}: New best model saved with Custom Score: {custom_score:.4f}"
                )

                # Update global top models
                update_top_models(
                    model_state_dict=model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict(),
                    custom_score=custom_score,
                    trial_number=trial.number,
                    fold_number=fold +1,
                )

        fold_scores.append(best_fold_score)
        logging.info(f"Fold {fold +1} Best Custom Score: {best_fold_score:.4f}")

    # Calculate average score across folds
    avg_score = np.mean(fold_scores)
    logging.info(f"Trial {trial.number}: Average Custom Score across folds: {avg_score:.4f}")

    return avg_score

# ===============================
# 12. Run Optuna Study
# ===============================

# To utilize multiple GPUs, Optuna's study should run in a single process.
# Ensure that the DataParallel setup is correctly applied within the objective function.

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=9, timeout=None)  # Increased n_trials for better hyperparameter exploration

logging.info("===== Best Trial =====")
best_trial = study.best_trial

logging.info(f"  Value (Average Custom Score): {best_trial.value}")
logging.info("  Params: ")
for key, value in best_trial.params.items():
    logging.info(f"    {key}: {value}")

# ===============================
# 13. Save Top 3 Models
# ===============================

def save_top_models(top_models):
    """
    Saves the top models to the checkpoints directory.

    Args:
        top_models (list): List of top model dictionaries.
    """
    os.makedirs("checkpoints/top_models", exist_ok=True)
    for idx, model_info in enumerate(top_models):
        save_path = f"checkpoints/top_models/best_model_{idx +1}_score_{model_info['score']:.4f}_trial_{model_info['trial']}_fold_{model_info['fold']}.pth"
        torch.save(model_info["state_dict"], save_path)
        logging.info(f"Saved Top {idx +1} Model: {save_path}")

    # Optionally, save the top_models information to a JSON file
    top_models_info = [
        {
            "score": model["score"],
            "trial": model["trial"],
            "fold": model["fold"],
            "path": f"checkpoints/top_models/best_model_{idx +1}_score_{model['score']:.4f}_trial_{model['trial']}_fold_{model['fold']}.pth",
        }
        for idx, model in enumerate(top_models)
    ]
    with open("checkpoints/top_models/top_models_info.json", "w") as f:
        json.dump(top_models_info, f, indent=4)
    logging.info("Top models information saved to 'checkpoints/top_models/top_models_info.json'.")

# Save the top 3 models after the study
save_top_models(top_models)

# ===============================
# 14. Final Remarks
# ===============================

logging.info("Training and Hyperparameter Tuning Completed.")
logging.info("Top 3 models have been saved in 'checkpoints/top_models/' directory.")
