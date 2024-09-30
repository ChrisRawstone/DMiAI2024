# train_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import ViT_B_16_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.make_dataset import LoadTifDataset

import random
import logging

from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Optuna for hyperparameter tuning
import optuna

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
logging.info(f"Number of GPUs available: {num_gpus}")

# Create necessary directories
os.makedirs('plots', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ===============================
# 2. Custom Score Function
# ===============================

def calculate_custom_score(y_true, y_pred):
    """
    Calculates the custom score based on the formula:
    Score = (a0 * a1) / (n0 * n1)

    Where:
    - a0: True Negatives (correctly predicted as 0)
    - a1: True Positives (correctly predicted as 1)
    - n0: Total actual class 0 samples
    - n1: Total actual class 1 samples
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Negatives (a0)
    a0 = np.sum((y_true == 0) & (y_pred == 0))

    # True Positives (a1)
    a1 = np.sum((y_true == 1) & (y_pred == 1))

    # Total actual class 0 and class 1
    n0 = np.sum(y_true == 0)
    n1 = np.sum(y_true == 1)

    # Avoid division by zero
    if n0 == 0 or n1 == 0:
        logging.warning("One of the classes has zero samples. Returning score as 0.")
        return 0.0

    score = (a0 * a1) / (n0 * n1)
    return score

# ===============================
# 4. Get Data and Dataloaders
# ===============================

def get_dataloaders(batch_size, img_size):
    """
    Returns DataLoader objects for training and validation datasets.

    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (int): Image size for resizing.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    # Define transformations
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                    std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                    std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
        ToTensorV2(),
    ])

    # Paths to images and CSV files
    image_dir = "data/training"
    csv_file_path = "data/training.csv"

    image_dir_val = "data/validation16bit"
    csv_file_path_val = "data/validation.csv"

    # Create the datasets
    train_dataset = LoadTifDataset(image_dir=image_dir, csv_file_path=csv_file_path, transform=train_transform)
    val_dataset = LoadTifDataset(image_dir=image_dir_val, csv_file_path=csv_file_path_val, transform=val_transform)

    # Extract labels from the dataset
    train_labels = train_dataset.labels_df.iloc[:, 1].values  # assuming labels are in the second column

    # Compute class counts and weights
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    samples_weights = class_weights[train_labels]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    batch_size = 16
    img_size = 224
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, img_size=img_size)
    print(train_loader.dataset)
    # (Initialize your model, optimizer, loss function, etc.)

    # (Training loop)


# # ===============================
# # 8. Save Sample Images
# # ===============================

# def save_sample_images(dataset, num_samples=5, folder='plots', split='train'):
#     """
#     Saves a few sample images from the dataset to verify correctness.

#     Args:
#         dataset (CustomDataset): The dataset to sample from.
#         num_samples (int, optional): Number of samples to save. Defaults to 5.
#         folder (str, optional): Directory to save images. Defaults to 'plots'.
#         split (str, optional): Dataset split name ('train' or 'val'). Defaults to 'train'.
#     """
#     os.makedirs(folder, exist_ok=True)
#     for i in range(num_samples):
#         image, label = dataset[i]
#         image_np = image.permute(1, 2, 0).cpu().numpy()
#         image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
#         image_np = np.clip(image_np, 0, 1)
#         plt.imshow(image_np)
#         plt.title(f'{split.capitalize()} Label: {label}')
#         plt.axis('off')
#         plt.savefig(f'{folder}/{split}_sample_{i}.png')
#         plt.close()

# # Uncomment the following lines if you want to save sample images
# # train_loader_temp, val_loader_temp = get_dataloaders(batch_size=1, img_size=224)
# # save_sample_images(train_loader_temp.dataset, split='train')
# # save_sample_images(val_loader_temp.dataset, split='val')

# logging.info("Sample images saved to 'plots' folder.")

# # ===============================
# # 9. Define Models Dictionary
# # ===============================

# def get_models(num_classes=1):
#     """
#     Returns a dictionary of state-of-the-art models.

#     Args:
#         num_classes (int, optional): Number of output classes. Defaults to 1.

#     Returns:
#         dict: Dictionary of models.
#     """
#     models_dict = {}

#     # Vision Transformer
#     vit_weights = ViT_B_16_Weights.DEFAULT
#     vit_model = models.vit_b_16(weights=vit_weights)
#     in_features = vit_model.heads.head.in_features
#     vit_model.heads.head = nn.Linear(in_features, num_classes)
#     models_dict['ViT'] = vit_model

#     # ResNet50
#     resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#     in_features = resnet_model.fc.in_features
#     resnet_model.fc = nn.Linear(in_features, num_classes)
#     models_dict['ResNet50'] = resnet_model

#     # EfficientNet
#     efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
#     in_features = efficientnet_model.classifier[1].in_features
#     efficientnet_model.classifier[1] = nn.Linear(in_features, num_classes)
#     models_dict['EfficientNet'] = efficientnet_model

#     # MobileNetV3
#     mobilenet_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
#     in_features = mobilenet_model.classifier[3].in_features
#     mobilenet_model.classifier[3] = nn.Linear(in_features, num_classes)
#     models_dict['MobileNetV3'] = mobilenet_model

#     return models_dict

# # ===============================
# # 10. Define Focal Loss with Class Weights
# # ===============================

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
#         """
#         Initializes the Focal Loss function.

#         Args:
#             alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
#             gamma (float, optional): Focusing parameter for modulating factor (1-p). Defaults to 2.
#             logits (bool, optional): Whether inputs are logits. Defaults to True.
#             reduce (bool, optional): Whether to reduce the loss. Defaults to True.
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         """
#         Forward pass for Focal Loss.

#         Args:
#             inputs (torch.Tensor): Predicted logits.
#             targets (torch.Tensor): Ground truth labels.

#         Returns:
#             torch.Tensor: Computed Focal Loss.
#         """
#         targets = targets.type_as(inputs)
#         if self.logits:
#             BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         else:
#             BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss

# # ===============================
# # 11. Hyperparameter Tuning with Optuna
# # ===============================

# def get_model_parallel(model):
#     """
#     Wraps the model with DataParallel if multiple GPUs are available.

#     Args:
#         model (nn.Module): The model to wrap.

#     Returns:
#         nn.Module: The potentially parallelized model.
#     """
#     if torch.cuda.device_count() > 1:
#         logging.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
#         model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
#     else:
#         logging.info("Using a single GPU.")
#     return model

# def objective(trial):
#     """
#     Objective function for Optuna hyperparameter optimization.

#     Args:
#         trial (optuna.trial.Trial): Optuna trial object.

#     Returns:
#         float: Validation custom score to maximize.
#     """
#     # Hyperparameters to tune
#     model_name = trial.suggest_categorical('model_name', ['ViT', 'ResNet50', 'EfficientNet', 'MobileNetV3'])

#     if model_name == 'ViT':
#         img_size = 224  # Fixed for ViT
#     else:
#         img_size = trial.suggest_categorical('img_size', [224, 256, 299])

#     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
#     lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
#     weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
#     gamma = trial.suggest_float('gamma', 1.0, 3.0)
#     alpha = trial.suggest_float('alpha', 0.1, 0.9)

#     num_epochs = 10  # For quick experimentation; increase for better results

#     # Get model
#     models_dict = get_models()
#     model = models_dict[model_name]
#     model = get_model_parallel(model)
#     model = model.to(device)

#     # Get dataloaders
#     train_loader, val_loader = get_dataloaders(batch_size, img_size)

#     # Loss function
#     criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

#     # Optimizer and Scheduler
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

#     # Mixed Precision Scaler
#     scaler = torch.amp.GradScaler('cuda')  # Removed 'device_type' argument

#     best_custom_score = 0

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_targets = []

#         progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
#         for images, labels in progress_bar:
#             images = images.to(device, non_blocking=True)
#             labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#             optimizer.zero_grad()

#             with torch.amp.autocast("cuda"):  # Removed 'device_type' argument
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             train_loss += loss.item() * images.size(0)

#             preds = torch.sigmoid(outputs).detach().cpu().numpy()
#             train_preds.extend(preds)
#             train_targets.extend(labels.detach().cpu().numpy())

#             progress_bar.set_postfix({'Loss': loss.item()})

#         avg_train_loss = train_loss / len(train_loader.dataset)

#         # Validation
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_targets = []

#         with torch.no_grad():
#             progress_bar = tqdm(val_loader, desc='Validation', leave=False)
#             for images, labels in progress_bar:
#                 images = images.to(device, non_blocking=True)
#                 labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#                 with torch.amp.autocast('cuda'):  # Removed 'device_type' argument
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)

#                 val_loss += loss.item() * images.size(0)

#                 preds = torch.sigmoid(outputs).detach().cpu().numpy()
#                 val_preds.extend(preds)
#                 val_targets.extend(labels.detach().cpu().numpy())

#         avg_val_loss = val_loss / len(val_loader.dataset)

#         # Calculate custom score
#         preds_binary = (np.array(val_preds) > 0.5).astype(int)
#         custom_score = calculate_custom_score(val_targets, preds_binary)

#         scheduler.step()

#         # Update best score
#         if custom_score > best_custom_score:
#             best_custom_score = custom_score

#         # Report intermediate objective value
#         trial.report(custom_score, epoch)

#         # Handle pruning based on the intermediate value
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()

#     return best_custom_score

# # ===============================
# # 12. Run Optuna Study
# # ===============================

# # To utilize multiple GPUs, Optuna's study should run in a single process.
# # Ensure that the DataParallel setup is correctly applied within the objective function.

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20, timeout=None)

# logging.info("Best trial:")
# best_trial = study.best_trial

# logging.info(f"  Value: {best_trial.value}")
# logging.info("  Params: ")
# for key, value in best_trial.params.items():
#     logging.info(f"    {key}: {value}")

# # ===============================
# # 13. Train Best Model with Best Hyperparameters
# # ===============================

# def train_best_model(trial):
#     """
#     Trains the best model with the best hyperparameters found by Optuna.
    
#     Args:
#         trial (optuna.trial.FrozenTrial): Best trial object from Optuna study.
#     """
#     model_name = trial.params['model_name']

#     if model_name == 'ViT':
#         img_size = 224  # Fixed for ViT
#     else:
#         img_size = trial.params.get('img_size', 224)  # Default to 224 if not present

#     batch_size = trial.params['batch_size']
#     lr = trial.params['lr']
#     weight_decay = trial.params['weight_decay']
#     gamma = trial.params['gamma']
#     alpha = trial.params['alpha']

#     num_epochs = 50  # Increase epochs for final training

#     # Get model
#     models_dict = get_models()
#     model = models_dict[model_name]
#     model = get_model_parallel(model)
#     model = model.to(device)

#     # Get dataloaders
#     train_loader, val_loader = get_dataloaders(batch_size, img_size)

#     # Loss function
#     criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

#     # Optimizer and Scheduler
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

#     # Mixed Precision Scaler
#     scaler = torch.amp.GradScaler('cuda')  # Removed 'device_type' argument

#     best_custom_score = 0
#     early_stopping_patience = 10
#     patience_counter = 0

#     for epoch in range(num_epochs):
#         logging.info(f"Epoch {epoch+1}/{num_epochs}")
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_targets = []

#         progress_bar = tqdm(train_loader, desc='Training', leave=False)
#         for images, labels in progress_bar:
#             images = images.to(device, non_blocking=True)
#             labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#             optimizer.zero_grad()

#             with torch.amp.autocast('cuda'):  # Removed 'device_type' argument
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             train_loss += loss.item() * images.size(0)

#             preds = torch.sigmoid(outputs).detach().cpu().numpy()
#             train_preds.extend(preds)
#             train_targets.extend(labels.detach().cpu().numpy())

#             progress_bar.set_postfix({'Loss': loss.item()})

#         avg_train_loss = train_loss / len(train_loader.dataset)

#         # Validation
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_targets = []

#         with torch.no_grad():
#             progress_bar = tqdm(val_loader, desc='Validation', leave=False)
#             for images, labels in progress_bar:
#                 images = images.to(device, non_blocking=True)
#                 labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#                 with torch.amp.autocast('cuda'):  # Removed 'device_type' argument
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)

#                 val_loss += loss.item() * images.size(0)

#                 preds = torch.sigmoid(outputs).detach().cpu().numpy()
#                 val_preds.extend(preds)
#                 val_targets.extend(labels.detach().cpu().numpy())

#         avg_val_loss = val_loss / len(val_loader.dataset)

#         # Calculate accuracy
#         preds_binary = (np.array(val_preds) > 0.5).astype(int)
#         val_acc = accuracy_score(val_targets, preds_binary) * 100  # Convert to percentage

#         # Calculate custom score
#         custom_score = calculate_custom_score(val_targets, preds_binary)

#         logging.info(f"Train Loss: {avg_train_loss:.4f}")
#         logging.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

#         # Scheduler step
#         scheduler.step()

#         # Checkpointing based on custom score
#         if custom_score > best_custom_score:
#             best_custom_score = custom_score
#             # Save the underlying model's state_dict
#             if isinstance(model, nn.DataParallel):
#                 torch.save(model.module.state_dict(), 'checkpoints/best_model.pth')
#             else:
#                 torch.save(model.state_dict(), 'checkpoints/best_model.pth')
#             logging.info(f"Best model saved with Val Score: {best_custom_score:.4f}")
#             patience_counter = 0

#             # Save model architecture name
#             model_info = {
#                 'model_name': model_name,
#                 'img_size': img_size
#             }
#             with open('checkpoints/model_info.json', 'w') as f:
#                 json.dump(model_info, f)
#             logging.info("Model architecture information saved to 'checkpoints/model_info.json'.")

#         else:
#             patience_counter += 1
#             logging.info(f"No improvement in Val Score for {patience_counter} epochs.")

#         # Early Stopping
#         if patience_counter >= early_stopping_patience:
#             logging.info("Early stopping triggered.")
#             break

#     # Load best model
#     if isinstance(model, nn.DataParallel):
#         model.module.load_state_dict(torch.load('checkpoints/best_model.pth'))
#     else:
#         model.load_state_dict(torch.load('checkpoints/best_model.pth'))
#     model.eval()

#     # Final Evaluation
#     val_loss = 0
#     val_preds = []
#     val_targets = []

#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc='Final Evaluation', leave=False)
#         for images, labels in progress_bar:
#             images = images.to(device, non_blocking=True)
#             labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#             with torch.amp.autocast('cuda'):  # Removed 'device_type' argument
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#             val_loss += loss.item() * images.size(0)

#             preds = torch.sigmoid(outputs).detach().cpu().numpy()
#             val_preds.extend(preds)
#             val_targets.extend(labels.detach().cpu().numpy())

#     avg_val_loss = val_loss / len(val_loader.dataset)
#     preds_binary = (np.array(val_preds) > 0.5).astype(int)
#     val_acc = accuracy_score(val_targets, preds_binary) * 100
#     custom_score = calculate_custom_score(val_targets, preds_binary)

#     logging.info("===== Final Evaluation =====")
#     logging.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

#     # Save the final model
#     if isinstance(model, nn.DataParallel):
#         torch.save(model.module.state_dict(), 'checkpoints/final_model.pth')
#     else:
#         torch.save(model.state_dict(), 'checkpoints/final_model.pth')
#     logging.info("Final model saved as 'checkpoints/final_model.pth'.")
    
#     # Optionally, save the model architecture info for the final model
#     with open('checkpoints/final_model_info.json', 'w') as f:
#         json.dump(model_info, f)
#     logging.info("Final model architecture information saved to 'checkpoints/final_model_info.json'.")

# # ===============================
# # 14. Execute Training of Best Model
# # ===============================

# # Train the best model
# train_best_model(best_trial)
