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

from src.utils import calculate_custom_score
from src.train_utils import FocalLoss, set_seed, save_sample_images

# ===============================
# 1. Important Parameters and Configuration
# ===============================

# Configuration Parameters
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

# Data Directories
DATA_DIR_TRAIN = "data/training"
CSV_FILE_TRAIN = "data/training.csv"
DATA_DIR_ORIG_VAL = "data/validation16bit"
CSV_FILE_ORIG_VAL = "data/validation.csv"

# Training Settings
NUM_EPOCHS = 50  # For quick experimentation; increase for better results
N_SPLITS = 5    # Number of folds for cross-validation
N_TRIALS = 50    # Number of Optuna trials
BATCH_SIZE_OPTIONS = [4, 8, 16, 32, 64, 128]
IMG_SIZE_OPTIONS = [128, 224, 256, 299, 331, 350, 400, 500]
MODEL_NAMES = [
    "ViT16",
    "ViT32",
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "EfficientNetB0",
    "EfficientNetB4",
    "MobileNetV3",
    "DenseNet121",
    "SwinV2B",  # Added SwinV2B to model names
]

# AMP Configuration
AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. Setup and Configuration
# ===============================

set_seed(SEED)

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

logging.info(f"Number of GPUs available: {NUM_GPUS}")

# Create directories
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints/top_models", exist_ok=True)       # For top 3 models from Optuna
os.makedirs("checkpoints/final_models", exist_ok=True)     # For final models trained on full data
os.makedirs("logs", exist_ok=True)

# ===============================
# 3. Custom Score Function
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
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
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

    # Create the datasets
    full_train_dataset = LoadTifDataset(
        image_dir=DATA_DIR_TRAIN, csv_file_path=CSV_FILE_TRAIN, transform=train_transform
    )
    original_val_dataset = LoadTifDataset(
        image_dir=DATA_DIR_ORIG_VAL,
        csv_file_path=CSV_FILE_ORIG_VAL,
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


# ===============================
# 5. Define Models Dictionary
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

    elif model_name == "SwinV2B":  # Added SwinV2B
        swin_v2_b_weights = Swin_V2_B_Weights.DEFAULT
        model = models.swin_v2_b(weights=swin_v2_b_weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model


# ===============================
# 6. Hyperparameter Tuning with Optuna
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
TOP_K = 3
top_models = []

def update_top_models(model_state_dict, custom_score, trial_number, fold_number, model_name, img_size):
    """
    Updates the global top_models list with the new model if it is among the top_k.

    Args:
        model_state_dict (dict): The state dictionary of the model.
        custom_score (float): The custom score of the model.
        trial_number (int): The trial number from Optuna.
        fold_number (int): The fold number in cross-validation.
        model_name (str): Name of the model architecture.
        img_size (int): Image size used during training.
    """
    global top_models
    if len(top_models) < TOP_K:
        top_models.append(
            {
                "score": custom_score,
                "state_dict": model_state_dict,
                "trial": trial_number,
                "fold": fold_number,
                "model_name": model_name,
                "img_size": img_size,
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
                "model_name": model_name,
                "img_size": img_size,
            }
            top_models = sorted(top_models, key=lambda x: x["score"], reverse=True)

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization with cross-validation.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Average validation custom score across all folds.
    """
    # Hyperparameters to tune
    model_name = trial.suggest_categorical("model_name", MODEL_NAMES)

    # Determine image size based on model
    if model_name in ['ViT16', 'ViT32']:
        img_size = 224  # Fixed for ViT
    elif model_name == "SwinV2B":
        img_size = 255  # Fixed for SwinV2B
    else:
        img_size = trial.suggest_categorical("img_size", IMG_SIZE_OPTIONS)

    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Load full training dataset
    full_train_dataset = LoadTifDataset(
        image_dir=DATA_DIR_TRAIN,
        csv_file_path=CSV_FILE_TRAIN,
        transform=A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
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
        logging.info(f"Starting Fold {fold + 1}/{N_SPLITS}")

        # Create DataLoaders for this fold
        train_loader, val_loader, original_val_loader = get_dataloaders(
            batch_size=batch_size,
            img_size=img_size,
            fold_indices=(train_idx, val_idx),
        )

        # Initialize model
        model = get_models(model_name=model_name)
        model = get_model_parallel(model)
        model = model.to(DEVICE)

        # Define loss, optimizer, scheduler
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler()

        best_fold_score = 0

        for epoch in range(NUM_EPOCHS):
            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{NUM_EPOCHS}")

            # Training Phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            progress_bar = tqdm(train_loader, desc=f"Fold {fold +1} Training Epoch {epoch+1}", leave=False)
            for images, labels_batch in progress_bar:
                images = images.to(DEVICE, non_blocking=True)
                labels_batch = labels_batch.float().unsqueeze(1).to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=AMP_DEVICE):
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
                    images = images.to(DEVICE, non_blocking=True)
                    labels_batch = labels_batch.float().unsqueeze(1).to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(device_type=AMP_DEVICE):
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
                    images = images.to(DEVICE, non_blocking=True)
                    labels_batch = labels_batch.float().unsqueeze(1).to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(device_type=AMP_DEVICE):
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

            # Update best score and potentially update top_models
            if custom_score > best_fold_score:
                best_fold_score = custom_score

                # Update global top models
                update_top_models(
                    model_state_dict=model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict(),
                    custom_score=custom_score,
                    trial_number=trial.number,
                    fold_number=fold +1,
                    model_name=model_name,      # Pass model_name
                    img_size=img_size,          # Pass img_size
                )

        fold_scores.append(best_fold_score)
        logging.info(f"Fold {fold +1} Best Custom Score: {best_fold_score:.4f}")

    # Calculate average score across folds
    avg_score = np.mean(fold_scores)
    logging.info(f"Trial {trial.number}: Average Custom Score across folds: {avg_score:.4f}")

    return avg_score

# ===============================
# 7. Run Optuna Study
# ===============================

# To utilize multiple GPUs, Optuna's study should run in a single process.
# Ensure that the DataParallel setup is correctly applied within the objective function.

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, timeout=None)  # Increased n_trials for better hyperparameter exploration

logging.info("===== Best Trial =====")
best_trial = study.best_trial

logging.info(f"  Value (Average Custom Score): {best_trial.value}")
logging.info("  Params: ")
for key, value in best_trial.params.items():
    logging.info(f"    {key}: {value}")

# ===============================
# 8. Save Top 3 Models
# ===============================

def save_top_models(top_models):
    """
    Saves the top models to the checkpoints/top_models directory.

    Args:
        top_models (list): List of top model dictionaries.
    """
    for idx, model_info in enumerate(top_models):
        save_path = f"checkpoints/top_models/best_model_{idx +1}_score_{model_info['score']:.4f}_trial_{model_info['trial']}_fold_{model_info['fold']}.pth"
        torch.save(model_info["state_dict"], save_path)
        logging.info(f"Saved Top {idx +1} Model: {save_path}")

    # Save the top_models information to a JSON file
    top_models_info = [
        {
            "score": model["score"],
            "trial": model["trial"],
            "fold": model["fold"],
            "model_name": model["model_name"],
            "img_size": model["img_size"],
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
# 9. Train Top 3 Models on Full Training Data and Validate
# ===============================

def train_and_validate_top_models(top_models, num_epochs=NUM_EPOCHS):
    """
    Trains the top models on the full training dataset and validates on the original validation dataset.

    Args:
        top_models (list): List of top model dictionaries.
        num_epochs (int, optional): Number of epochs for training. Defaults to NUM_EPOCHS.
    """
    for idx, model_info in enumerate(top_models):
        logging.info(f"===== Training Top Model {idx +1} =====")
        model_score = model_info["score"]
        trial_number = model_info["trial"]
        fold_number = model_info["fold"]
        model_name = model_info["model_name"]
        img_size = model_info["img_size"]
        state_dict = model_info["state_dict"]

        # Retrieve hyperparameters from the study
        trial = study.trials[trial_number]
        params = trial.params

        # Extract hyperparameters with defaults
        batch_size = params.get("batch_size", 32)
        lr = params.get("lr", 1e-4)
        weight_decay = params.get("weight_decay", 1e-4)
        gamma = params.get("gamma", 2.0)
        alpha = params.get("alpha", 0.25)

        logging.info(
            f"Top Model {idx +1} - Trial {trial_number}, Fold {fold_number}, "
            f"Model: {model_name}, Img Size: {img_size}, Batch Size: {batch_size}, "
            f"LR: {lr}, Weight Decay: {weight_decay}, Gamma: {gamma}, Alpha: {alpha}"
        )

        # Initialize model
        model = get_models(model_name=model_name)
        model = get_model_parallel(model)
        model = model.to(DEVICE)

        # Load the state_dict
        model.load_state_dict(state_dict)

        # Define loss, optimizer, scheduler
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Mixed Precision Scaler
        scaler = torch.amp.GradScaler()

        # Create DataLoaders for full training data and original validation data
        full_train_loader, _, original_val_loader = get_dataloaders(
            batch_size=batch_size,
            img_size=img_size,
            fold_indices=None,  # Use entire training data
        )

        best_model_score = 0
        best_model_path = f"checkpoints/final_models/final_model_{idx +1}_score_{model_score:.4f}.pth"
        best_model_info_path = f"checkpoints/final_models/final_model_{idx +1}_info.json"

        for epoch in range(num_epochs):
            logging.info(f"Top Model {idx +1} - Epoch {epoch +1}/{num_epochs}")

            # Training Phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            progress_bar = tqdm(full_train_loader, desc=f"Top Model {idx +1} Training Epoch {epoch+1}", leave=False)
            for images, labels_batch in progress_bar:
                images = images.to(DEVICE, non_blocking=True)
                labels_batch = labels_batch.float().unsqueeze(1).to(DEVICE, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=AMP_DEVICE):
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

            avg_train_loss = train_loss / len(full_train_loader.dataset)

            # Validation Phase (Original Validation Set)
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                progress_bar = tqdm(original_val_loader, desc=f"Top Model {idx +1} Validation Epoch {epoch+1}", leave=False)
                for images, labels_batch in progress_bar:
                    images = images.to(DEVICE, non_blocking=True)
                    labels_batch = labels_batch.float().unsqueeze(1).to(DEVICE, non_blocking=True)

                    with torch.amp.autocast(device_type=AMP_DEVICE):
                        outputs = model(images)
                        loss = criterion(outputs, labels_batch)

                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels_batch.detach().cpu().numpy())

            avg_val_loss = val_loss / len(original_val_loader.dataset)

            # Calculate custom score
            preds_binary = (np.array(val_preds) > 0.5).astype(int)
            custom_score = calculate_custom_score(val_targets, preds_binary)

            # Log the scores
            logging.info(
                f"Top Model {idx +1} Epoch {epoch +1}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Custom Score (Original Val): {custom_score:.4f}"
            )

            # Update scheduler
            scheduler.step()

            # Save the best model based on custom score
            if custom_score > best_model_score:
                best_model_score = custom_score
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                logging.info(
                    f"Top Model {idx +1} Epoch {epoch +1}: New best model saved with Custom Score: {custom_score:.4f}"
                )

                # Save the model's info to a JSON file
                final_model_info = {
                    "model_name": model_name,
                    "img_size": img_size,
                    "score": custom_score,
                    "trial": trial_number,
                    "fold": fold_number,
                    "final_model_path": best_model_path,
                }
                with open(best_model_info_path, "w") as f:
                    json.dump(final_model_info, f, indent=4)
                logging.info(f"Saved Final Model Info {idx +1}: {best_model_info_path}")

        logging.info(
            f"Top Model {idx +1} Training Completed. Best Custom Score: {best_model_score:.4f}"
        )

# Train and validate the top 3 models
train_and_validate_top_models(top_models, num_epochs=NUM_EPOCHS)

# ===============================
# 10. Save Final Models Information
# ===============================

def save_final_models_info(top_models):
    """
    Saves the final models information to individual JSON files in the final_models directory.

    Args:
        top_models (list): List of top model dictionaries.
    """
    for idx, model_info in enumerate(top_models):
        final_model_path = f"checkpoints/final_models/final_model_{idx +1}_score_{model_info['score']:.4f}.pth"
        final_model_info = {
            "model_name": model_info["model_name"],
            "img_size": model_info["img_size"],
            "score": model_info["score"],
            "trial": model_info["trial"],
            "fold": model_info["fold"],
            "final_model_path": final_model_path,
        }

        # Save the model's state_dict
        torch.save(model_info["state_dict"], final_model_path)
        logging.info(f"Saved Final Model {idx +1}: {final_model_path}")

        # Save the model's info to a JSON file
        info_path = f"checkpoints/final_models/final_model_{idx +1}_info.json"
        with open(info_path, "w") as f:
            json.dump(final_model_info, f, indent=4)
        logging.info(f"Saved Final Model Info {idx +1}: {info_path}")

# Save the final models information
save_final_models_info(top_models)

# ===============================
# 11. Save Final Models Information (Alternative Consolidated)
# ===============================

def save_final_models_info_consolidated(top_models):
    """
    Saves the final models information to a single JSON file in the final_models directory.

    Args:
        top_models (list): List of top model dictionaries.
    """
    final_models_info = []
    for idx, model_info in enumerate(top_models):
        final_model_path = f"checkpoints/final_models/final_model_{idx +1}_score_{model_info['score']:.4f}.pth"
        info_path = f"checkpoints/final_models/final_model_{idx +1}_info.json"

        # Save the model's state_dict
        torch.save(model_info["state_dict"], final_model_path)
        logging.info(f"Saved Final Model {idx +1}: {final_model_path}")

        # Prepare the model's info
        model_details = {
            "model_name": model_info["model_name"],
            "img_size": model_info["img_size"],
            "score": model_info["score"],
            "trial": model_info["trial"],
            "fold": model_info["fold"],
            "final_model_path": final_model_path,
            "info_path": info_path,
        }

        # Save the model's info to an individual JSON file
        with open(info_path, "w") as f:
            json.dump(model_details, f, indent=4)
        logging.info(f"Saved Final Model Info {idx +1}: {info_path}")

        # Append to the consolidated list
        final_models_info.append(model_details)

    # Save the consolidated info
    consolidated_info_path = "checkpoints/final_models/final_models_consolidated_info.json"
    with open(consolidated_info_path, "w") as f:
        json.dump(final_models_info, f, indent=4)
    logging.info(f"Saved Consolidated Final Models Info: {consolidated_info_path}")

# Optionally, uncomment the following line to save a consolidated JSON file
# save_final_models_info_consolidated(top_models)

# ===============================
# 12. Final Remarks
# ===============================

logging.info("Training and Hyperparameter Tuning Completed.")
logging.info("Top 3 models have been saved in 'checkpoints/top_models/' directory.")
logging.info("Final models trained on the full training data have been saved in 'checkpoints/final_models/' directory.")
logging.info("All model information has been saved in the respective JSON files.")

# ===============================
# 13. Model Loading Function
# ===============================

def load_model(checkpoint_path, model_info_path, device):
    """
    Loads the model architecture and weights.

    Args:
        checkpoint_path (str): Path to the model weights.
        model_info_path (str): Path to the model architecture info JSON.
        device (torch.device): Device to load the model on.

    Returns:
        tuple: (model, img_size, model_info)
    """
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']

    model = get_models(model_name, num_classes=1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, img_size, model_info

# ===============================
# 14. Example Usage of load_model
# ===============================

# Example usage (uncomment and modify the paths as needed):
# checkpoint_path = "checkpoints/final_models/final_model_1_score_0.9500.pth"
# model_info_path = "checkpoints/final_models/final_model_1_info.json"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, img_size, model_info = load_model(checkpoint_path, model_info_path, device)

# Now, `model` is ready for inference or further training

