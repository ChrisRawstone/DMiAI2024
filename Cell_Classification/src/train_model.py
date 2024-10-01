# train_model.py

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ViT_B_32_Weights, ResNet18_Weights, ViT_B_16_Weights, ResNet50_Weights, ResNet101_Weights, EfficientNet_B0_Weights, EfficientNet_B4_Weights, MobileNet_V3_Large_Weights
from data.make_dataset import LoadTifDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import random
import logging
import gc

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import logging

# Optuna for hyperparameter tuning
import optuna


from src.utils import calculate_custom_score

wandb.login(key="c187178e0437c71d461606e312d20dc9f1c6794f")

# ===============================
# 1. Setup and Configuration
# ===============================

SEED = 0

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Define device and check available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

# Setup logging early to capture GPU info
os.makedirs('logs', exist_ok=True)
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

logging.info(f"Number of GPUs available: {num_gpus}")


# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ===============================
# 2. Custom Score Function
# ===============================


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
        # A.RandomBrightnessContrast(p=0.5),
        # A.RandomGamma(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means  alternative : A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds  alternative : std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.485, 0.485),  # Using ImageNet means
                    std=(0.229, 0.229, 0.229)),   # Using ImageNet stds
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

    # Compute class counts and total samples
    class_counts = train_dataset.labels_df['is_homogenous'].value_counts().to_dict()
    total_samples = len(train_labels)
    
    # Initialize weights list
    weights = [1.0] * total_samples

    # Assign weights to each sample
    for idx, label in enumerate(train_labels):
        if label == 0:
            weights[idx] = total_samples / (2 * class_counts.get(0, 1))
        else:
            weights[idx] = total_samples / (2 * class_counts.get(1, 1))

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)

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

def save_sample_images(dataset, num_samples=5, folder='plots', split='train'):
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
        image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
        image_np = np.clip(image_np, 0, 1)
        plt.imshow(image_np)
        plt.title(f'{split.capitalize()} Label: {label}')
        plt.axis('off')
        plt.savefig(f'{folder}/{split}_sample_{i}.png')
        plt.close()

# Uncomment the following lines if you want to save sample images
# train_loader_temp, val_loader_temp = get_dataloaders(batch_size=1, img_size=224)
# save_sample_images(train_loader_temp.dataset, split='train')
# save_sample_images(val_loader_temp.dataset, split='val')

# logging.info("Sample images saved to 'plots' folder.")

# ===============================
# 9. Define Models Dictionary
# ===============================

def get_models(num_classes=1):
    """
    Returns a dictionary of state-of-the-art models.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1.

    Returns:
        dict: Dictionary of models.
    """
    models_dict = {}

    # Vision Transformer16
    vit_weights = ViT_B_16_Weights.DEFAULT
    vit_model = models.vit_b_16(weights=vit_weights)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT16'] = vit_model
    
    # # Vision Transformer16
    # vit_weights_h14 = ViT_H_14_Weights.DEFAULT
    # vit_model14 = models.vit_h_14(weights=vit_weights_h14)
    # in_features = vit_model14.heads.head.in_features
    # vit_model14.heads.head = nn.Linear(in_features, num_classes)
    # models_dict['ViTh14'] = vit_model14
    
    # Vision Transformer32
    vit_weights32 = ViT_B_32_Weights.DEFAULT
    vit_model32 = models.vit_b_32(weights=vit_weights32)
    in_features = vit_model32.heads.head.in_features
    vit_model32.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT32'] = vit_model32

    # ResNet18
    resnet18_weights = ResNet18_Weights.DEFAULT
    resnet18_model = models.resnet18(weights=resnet18_weights)
    in_features = resnet18_model.fc.in_features
    resnet18_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet18'] = resnet18_model

    # ResNet50
    resnet50_weights = ResNet50_Weights.DEFAULT
    resnet50_model = models.resnet50(weights=resnet50_weights)
    in_features = resnet50_model.fc.in_features
    resnet50_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet50'] = resnet50_model

    # ResNet101
    resnet101_weights = ResNet101_Weights.DEFAULT
    resnet101_model = models.resnet101(weights=resnet101_weights)
    in_features = resnet101_model.fc.in_features
    resnet101_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet101'] = resnet101_model

    # EfficientNetB0
    efficientnet_b0_weights = EfficientNet_B0_Weights.DEFAULT
    efficientnet_b0_model = models.efficientnet_b0(weights=efficientnet_b0_weights)
    in_features = efficientnet_b0_model.classifier[1].in_features
    efficientnet_b0_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB0'] = efficientnet_b0_model

    # EfficientNetB4
    efficientnet_b4_weights = EfficientNet_B4_Weights.DEFAULT
    efficientnet_b4_model = models.efficientnet_b4(weights=efficientnet_b4_weights)
    in_features = efficientnet_b4_model.classifier[1].in_features
    efficientnet_b4_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNetB4'] = efficientnet_b4_model

    # MobileNetV3 Large
    mobilenet_v3_weights = MobileNet_V3_Large_Weights.DEFAULT
    mobilenet_v3_model = models.mobilenet_v3_large(weights=mobilenet_v3_weights)
    in_features = mobilenet_v3_model.classifier[3].in_features
    mobilenet_v3_model.classifier[3] = nn.Linear(in_features, num_classes)
    models_dict['MobileNetV3'] = mobilenet_v3_model

    # DenseNet121
    densenet_weights = models.DenseNet121_Weights.DEFAULT
    densenet_model = models.densenet121(weights=densenet_weights)
    in_features = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(in_features, num_classes)
    models_dict['DenseNet121'] = densenet_model

    return models_dict


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
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
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

global best_custom_score
best_custom_score = 0

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization with wandb integration and early stopping.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation custom score to maximize.
    """
    torch.cuda.empty_cache()  # Clears the memory cache on the GPU
    gc.collect()  # Collects unreferenced objects in Python memory

    global best_custom_score


    # ---------------------------
    # 1. Hyperparameter Sampling
    # ---------------------------
    model_name = trial.suggest_categorical(
        'model_name',
        ['ViT16', 'ResNet50', "ResNet18", 'EfficientNetB0', 'MobileNetV3',  "EfficientNetB4",
         "ResNet101", "ViT32", 'DenseNet121']
    )

    if model_name in ['ViT16', "ViT32"]:
        img_size = 224  # Fixed for ViT
    else:
        img_size = trial.suggest_categorical(
            'img_size',
            [224, 400, 600, 800, 1000]
        )

    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)

    num_epochs = 100  # Total number of epochs
    patience = 20    # Early stopping patience
    


    # ---------------------------
    # 2. Initialize wandb Run
    # ---------------------------
    wandb_config = {
        'model_name': model_name,
        'img_size': img_size,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'alpha': alpha,
        'num_epochs': num_epochs,
        'patience': patience
    }




    wandb.init(
        project='Cell_Classification',  # Replace with your wandb project name
        config=wandb_config,
        reinit=True,  # Allows multiple wandb runs in the same script
        name=f"trial_{trial.number}"
    )



    # ---------------------------
    # 3. Model and Data Setup
    # ---------------------------
    models_dict = get_models()
    model = models_dict[model_name]
    model = get_model_parallel(model)
    model = model.to(device)

    train_loader, val_loader = get_dataloaders(batch_size, img_size)

    criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

    # ---------------------------
    # 3. Optimizer setup
    # ---------------------------

    optimizer_name = trial.suggest_categorical('optimizer_name', ['AdamW', 'SGD', 'RMSprop'])

    # Add specific configurations based on the optimizer chosen
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW' or optimizer_name == 'Adam':
        beta1 = trial.suggest_float('beta1', 0.8, 0.99)
        beta2 = trial.suggest_float('beta2', 0.9, 0.999)
        epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        alpha = trial.suggest_float('alpha', 0.9, 0.99)
        epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6)
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=epsilon, weight_decay=weight_decay)


    wandb_config.update({
    'optimizer_name': optimizer_name,
    'momentum': momentum if optimizer_name == 'SGD' else None,
    'beta1': beta1 if optimizer_name in ['AdamW', 'Adam'] else None,
    'beta2': beta2 if optimizer_name in ['AdamW', 'Adam'] else None,
    'epsilon': epsilon if optimizer_name in ['AdamW', 'Adam', 'RMSprop'] else None,
    'alpha': alpha if optimizer_name == 'RMSprop' else None
    })

    # ---------------------------
    # 3. Optimizer setup end
    # ---------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = torch.amp.GradScaler("cuda")  # For mixed precision training

    # running model with this configuration
    logging.info("###############################################")
    logging.info(f"Running model: {model_name} with Image Size: {img_size}, Batch Size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}, Gamma: {gamma}, Alpha: {alpha}")
    logging.info(f"with optimizer setting: {optimizer_name}")
    logging.info("###############################################")

    # ---------------------------
    # 4. Training with Early Stopping
    # ---------------------------
    trial_best_score = -np.inf
    epochs_without_improvement = 0

    try:
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")

            # Training Phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):  # Mixed precision
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(labels.detach().cpu().numpy())

                progress_bar.set_postfix({'Loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader.dataset)

            # Validation Phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc='Validation', leave=False)
                for images, labels in progress_bar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):  # Mixed precision
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels.detach().cpu().numpy())

            avg_val_loss = val_loss / len(val_loader.dataset)

            # Calculate custom score
            preds_binary = (np.array(val_preds) > 0.5).astype(int)
            custom_score = calculate_custom_score(val_targets, preds_binary)

            scheduler.step()

            # ---------------------------
            # 5. Logging Metrics to wandb
            # ---------------------------
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'custom_score': custom_score,
                'learning_rate': scheduler.get_last_lr()[0]
            })

            # ---------------------------
            # 6. Early Stopping Logic
            # ---------------------------
            if custom_score > trial_best_score:
                trial_best_score = custom_score
                epochs_without_improvement = 0

                # Update global best score and save model locally
                if custom_score > best_custom_score:
                    best_custom_score = custom_score

                    # Save the best model locally
                    save_path_optuna = 'checkpoints/best_model_optuna.pth'
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), save_path_optuna)
                    else:
                        torch.save(model.state_dict(), save_path_optuna)
                    logging.info(f"New best Optuna model saved with Val Score: {best_custom_score:.4f}")

                    # Save model hyperparameters locally
                    model_info_optuna = {
                        'model_name': model_name,
                        'img_size': img_size,
                        'batch_size': batch_size,
                        'learning_rate': lr,
                        'weight_decay': weight_decay,
                        'gamma': gamma,
                        'alpha': alpha,
                        'custom_score' : custom_score
                    }
                    with open('checkpoints/model_info_optuna.json', 'w') as f:
                        json.dump(model_info_optuna, f, indent=4)
            else:
                epochs_without_improvement += 1

            # Check if early stopping condition is met
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

            # ---------------------------
            # 7. Optuna Trial Reporting and Pruning
            # ---------------------------
            trial.report(custom_score, epoch)

            if trial.should_prune():
                logging.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

    except optuna.exceptions.TrialPruned:
        wandb.log({'pruned': True})
        wandb.finish(quiet=True)
        raise

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        wandb.finish(quiet=True)
        raise

    wandb.finish()

    torch.cuda.empty_cache()  # Clears the memory cache on the GPU
    gc.collect()  # Collects unreferenced objects in Python memory

    return best_custom_score



# ===============================
# 12. Run Optuna Study
# ===============================

# To utilize multiple GPUs, Optuna's study should run in a single process.
# Ensure that the DataParallel setup is correctly applied within the objective function.

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, timeout=None)  # Increased n_trials for better hyperparameter exploration

logging.info("===== Best Trial =====")
best_trial = study.best_trial

logging.info(f"  Value (Custom Score): {best_trial.value}")
logging.info("  Params: ")
for key, value in best_trial.params.items():
    logging.info(f"    {key}: {value}")



