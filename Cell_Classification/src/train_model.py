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
from torchvision.models import ViT_B_16_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
import logging

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import logging

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
# 3. Data Decoding Functions
# ===============================

# (No changes needed here as duplicate imports were removed)

# ===============================
# 4. Custom Dataset
# ===============================

class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx]['image_id']).zfill(3)
        label = int(self.df.iloc[idx]['is_homogenous'])
        image_path = os.path.join(self.image_dir, f'{image_id}.tif')

        # Read image directly using cv2
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logging.error(f"Image file not found or failed to load: {image_path}")
            # Return a black image if file not found
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            label = 0
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = transforms.ToTensor()(image)
            return image, label

        # Handle 16-bit images by converting to 8-bit
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # Handle images with alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = transforms.ToTensor()(image).permute(2, 0, 1)

        return image, label

# ===============================
# 5. Data Augmentation and Transforms
# ===============================

def get_transforms(img_size=224):
    """
    Returns training and validation transforms.

    Args:
        img_size (int, optional): Image size for resizing. Defaults to 224.

    Returns:
        tuple: Training and validation transforms.
    """
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

    return train_transform, val_transform

# ===============================
# 6. Load CSV Files
# ===============================

train_df = pd.read_csv('data/training.csv')
val_df = pd.read_csv('data/validation.csv')

logging.info(f"Training samples: {len(train_df)}")
logging.info(f"Validation samples: {len(val_df)}")

# ===============================
# 7. Create Datasets and DataLoaders with Weighted Sampling
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
    train_transform, val_transform = get_transforms(img_size=img_size)
    train_dataset = CustomDataset(train_df, 'data/training', transform=train_transform)
    val_dataset = CustomDataset(val_df, 'data/validation', transform=val_transform)

    # Calculate class weights for WeightedRandomSampler
    class_counts = train_df['is_homogenous'].value_counts().to_dict()
    total_samples = len(train_df)
    weights = [1.0] * len(train_df)

    for idx, label in enumerate(train_df['is_homogenous']):
        if label == 0:
            weights[idx] = total_samples / (2 * class_counts.get(0, 1))
        else:
            weights[idx] = total_samples / (2 * class_counts.get(1, 1))

    sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler for weighted sampling
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

# ===============================
# 8. Save Sample Images
# ===============================

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

    # Vision Transformer
    vit_weights = ViT_B_16_Weights.DEFAULT
    vit_model = models.vit_b_16(weights=vit_weights)
    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = nn.Linear(in_features, num_classes)
    models_dict['ViT'] = vit_model

    # ResNet50
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(in_features, num_classes)
    models_dict['ResNet50'] = resnet_model

    # EfficientNet
    efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(in_features, num_classes)
    models_dict['EfficientNet'] = efficientnet_model

    # MobileNetV3
    mobilenet_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = mobilenet_model.classifier[3].in_features
    mobilenet_model.classifier[3] = nn.Linear(in_features, num_classes)
    models_dict['MobileNetV3'] = mobilenet_model

    return models_dict

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

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation custom score to maximize.
    """
    # Hyperparameters to tune
    model_name = trial.suggest_categorical('model_name', ['ViT', 'ResNet50', 'EfficientNet', 'MobileNetV3'])

    if model_name == 'ViT':
        img_size = 224  # Fixed for ViT
    else:
        img_size = trial.suggest_categorical('img_size', [224, 256, 299])

    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)

    num_epochs = 10  # For quick experimentation; increase for better results

    # Get model
    models_dict = get_models()
    model = models_dict[model_name]
    model = get_model_parallel(model)
    model = model.to(device)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size, img_size)

    # Loss function
    criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")  # Updated to use torch.cuda.amp

    best_custom_score = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):  # Updated autocast usage
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

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):  # Updated autocast usage
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

        # Update best score and save model + JSON
        if custom_score > best_custom_score:
            best_custom_score = custom_score

            # Save the best model from Optuna hypertuning
            save_path_optuna = 'checkpoints/best_model_optuna.pth'
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path_optuna)
            else:
                torch.save(model.state_dict(), save_path_optuna)
            logging.info(f"New best Optuna model saved with Val Score: {best_custom_score:.4f}")

            # Save model architecture and hyperparameters for Optuna best model
            model_info_optuna = {
                'model_name': model_name,
                'img_size': img_size,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'gamma': gamma,
                'alpha': alpha
            }
            with open('checkpoints/model_info_optuna.json', 'w') as f:
                json.dump(model_info_optuna, f, indent=4)
            logging.info("Optuna best model architecture and hyperparameters saved to 'checkpoints/model_info_optuna.json'.")

        # Report intermediate objective value
        trial.report(custom_score, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_custom_score



# ===============================
# 12. Run Optuna Study
# ===============================

# To utilize multiple GPUs, Optuna's study should run in a single process.
# Ensure that the DataParallel setup is correctly applied within the objective function.

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1, timeout=None)  # Increased n_trials for better hyperparameter exploration

logging.info("===== Best Trial =====")
best_trial = study.best_trial

logging.info(f"  Value (Custom Score): {best_trial.value}")
logging.info("  Params: ")
for key, value in best_trial.params.items():
    logging.info(f"    {key}: {value}")

# ===============================
# 13. Train Best Model with Best Hyperparameters
# ===============================

def train_best_model(trial):
    """
    Trains the best model with the best hyperparameters found by Optuna.

    Args:
        trial (optuna.trial.FrozenTrial): Best trial object from Optuna study.
    """
    model_name = trial.params['model_name']

    if model_name == 'ViT':
        img_size = 224  # Fixed for ViT
    else:
        img_size = trial.params.get('img_size', 224)  # Default to 224 if not present

    batch_size = trial.params['batch_size']
    lr = trial.params['lr']
    weight_decay = trial.params['weight_decay']
    gamma = trial.params['gamma']
    alpha = trial.params['alpha']

    num_epochs = 5  # Increased epochs for final training

    # Get model
    models_dict = get_models()
    model = models_dict[model_name]
    model = get_model_parallel(model)
    model = model.to(device)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size, img_size)

    # Loss function
    criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda")  # Updated to use torch.cuda.amp

    best_custom_score = 0
    early_stopping_patience = 10
    patience_counter = 0

    # Initialize model_info dictionary
    model_info = {
        'model_name': model_name,
        'img_size': img_size,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'alpha': alpha
    }

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):  # Updated autocast usage
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

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):  # Updated autocast usage
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)

        # Calculate accuracy
        preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_acc = accuracy_score(val_targets, preds_binary) * 100  # Convert to percentage

        # Calculate custom score
        custom_score = calculate_custom_score(val_targets, preds_binary)

        logging.info(f"Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

        # Scheduler step
        scheduler.step()

        # Checkpointing based on custom score
        if custom_score > best_custom_score:
            best_custom_score = custom_score
            # Save the best model from final training with a different name
            save_path_final = 'checkpoints/best_model_final.pth'  # Changed filename
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path_final)
            else:
                torch.save(model.state_dict(), save_path_final)
            logging.info(f"Best final model saved with Val Score: {best_custom_score:.4f}")
            patience_counter = 0

            # Save model architecture name
            model_info_current = {
                'model_name': model_name,
                'img_size': img_size
            }
            with open('checkpoints/model_info_final.json', 'w') as f:
                json.dump(model_info_current, f)
            logging.info("Final model architecture information saved to 'checkpoints/model_info_final.json'.")

        else:
            patience_counter += 1
            logging.info(f"No improvement in Val Score for {patience_counter} epochs.")

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    # Load best model
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load('checkpoints/best_model_final.pth'))
    else:
        model.load_state_dict(torch.load('checkpoints/best_model_final.pth'))
    model.eval()

    # Final Evaluation
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Final Evaluation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):  # Updated autocast usage
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    val_acc = accuracy_score(val_targets, preds_binary) * 100
    custom_score = calculate_custom_score(val_targets, preds_binary)

    logging.info("===== Final Evaluation =====")
    logging.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

    # Save the final model
    save_path_final = 'checkpoints/final_model_final.pth'  # Changed filename to differentiate from Optuna best
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path_final)
    else:
        torch.save(model.state_dict(), save_path_final)
    logging.info(f"Final model saved as '{save_path_final}'.")

    # Optionally, save the model architecture info for the final model
    with open('checkpoints/final_model_info_final.json', 'w') as f:
        json.dump(model_info, f)
    logging.info("Final model architecture information saved to 'checkpoints/final_model_info_final.json'.")

    # ===============================
    # 15. Print Final Summary
    # ===============================

    logging.info("===== Best Model Configuration =====")
    for key, value in model_info.items():
        logging.info(f"{key}: {value}")

    logging.info("===== Final Evaluation Metrics =====")
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Validation Accuracy: {val_acc:.2f}%")
    logging.info(f"Custom Score: {custom_score:.4f}")

# ===============================
# 14. Execute Training of Best Model
# ===============================

# Train the best model
train_best_model(best_trial)
