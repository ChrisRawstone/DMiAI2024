import os
import cv2
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ViT_B_16_Weights  # Import the weights enum

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from PIL import Image

import random
import logging

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

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# 3. Data Decoding Functions
# ===============================

def decode_image(encoded_img: str) -> np.ndarray:
    """
    Decodes a base64 encoded image string to a NumPy array.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        np.ndarray: Decoded image.
    """
    try:
        img_data = base64.b64decode(encoded_img)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image decoding resulted in None.")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def load_sample(encoded_img: str) -> dict:
    """
    Loads and decodes the sample image.

    Args:
        encoded_img (str): Base64 encoded image string.

    Returns:
        dict: Dictionary containing the image.
    """
    image = decode_image(encoded_img)
    return {
        "image": image
    }

# ===============================
# 4. Custom Dataset
# ===============================

class CustomDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        Initializes the dataset with a DataFrame, image directory, and transformations.

        Args:
            df (pd.DataFrame): DataFrame containing 'image_id' and 'is_homogenous'.
            image_dir (str): Directory where images are stored.
            transform (albumentations.Compose, optional): Transformations to apply.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image tensor, label)
        """
        image_id = str(self.df.iloc[idx]['image_id']).zfill(3)
        label = int(self.df.iloc[idx]['is_homogenous'])
        image_path = os.path.join(self.image_dir, f'{image_id}.tif')

        # Read image file as binary
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            # Return a black image if file not found
            image = np.zeros((224, 224), dtype=np.uint8)
            label = 0
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = transforms.ToTensor()(image)
            return image, label

        # Encode image bytes to base64 string
        encoded_img = base64.b64encode(img_bytes).decode('utf-8')

        # Decode image using provided function
        image = decode_image(encoded_img)

        # Convert grayscale to RGB by duplicating channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = transforms.ToTensor()(image).permute(2, 0, 1)  # Convert to [C, H, W]

        return image, label

# ===============================
# 5. Data Augmentation and Transforms
# ===============================

train_transform = A.Compose([
    A.Resize(224, 224),
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
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),  # Using ImageNet means
                std=(0.229, 0.224, 0.225)),   # Using ImageNet stds
    ToTensorV2(),
])

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

train_dataset = CustomDataset(train_df, 'data/training8bit', transform=train_transform)
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
    batch_size=64,
    sampler=sampler,  # Use sampler for weighted sampling
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

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
# save_sample_images(train_dataset, split='train')
# save_sample_images(val_dataset, split='val')

logging.info("Sample images saved to 'plots' folder.")

# ===============================
# 9. Define the Model (Using Vision Transformer)
# ===============================

def get_model():
    """
    Initializes the Vision Transformer (ViT) model for binary classification.

    Returns:
        nn.Module: The modified ViT model.
    """
    # Initialize the Vision Transformer model with updated weights parameter
    weights = ViT_B_16_Weights.DEFAULT  # Use the default pretrained weights
    model = models.vit_b_16(weights=weights)
    
    # Check if model.heads is a Sequential or a single Linear layer
    if isinstance(model.heads, nn.Sequential):
        # Assuming the first layer is the Linear layer we want to modify
        in_features = model.heads[0].in_features
    else:
        in_features = model.heads.in_features
    
    # Replace the classification head for binary classification
    model.heads = nn.Linear(in_features, 1)
    
    return model

model = get_model()
model = model.to(device)

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

# Calculate class weights
class_counts = train_df['is_homogenous'].value_counts().to_dict()
total = sum(class_counts.values())
# Compute alpha for the minority class
alpha = class_counts.get(1, 1) / total  # Assuming 1 is the minority class
criterion = FocalLoss(alpha=alpha, gamma=2, logits=True).to(device)

# ===============================
# 11. Optimizer and Scheduler
# ===============================

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ===============================
# 12. Mixed Precision Scaler
# ===============================

scaler = torch.cuda.amp.GradScaler()

# ===============================
# 13. Training and Validation Loops with Custom Score
# ===============================

num_epochs = 50
best_custom_score = 0
early_stopping_patience = 10
patience_counter = 0

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

        with torch.cuda.amp.autocast():
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

    avg_train_loss = train_loss / len(train_dataset)
    try:
        train_auc = roc_auc_score(train_targets, train_preds)
    except ValueError:
        train_auc = 0.0  # Handle cases where AUC cannot be computed

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

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_dataset)
    try:
        val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        val_auc = 0.0  # Handle cases where AUC cannot be computed

    # Calculate accuracy
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    val_acc = accuracy_score(val_targets, preds_binary) * 100  # Convert to percentage

    # Calculate custom score
    custom_score = calculate_custom_score(val_targets, preds_binary)

    logging.info(f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")
    logging.info(f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

    # Scheduler step
    scheduler.step()

    # Checkpointing based on custom score
    if custom_score > best_custom_score:
        best_custom_score = custom_score
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        logging.info(f"Best model saved with Val Score: {best_custom_score:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        logging.info(f"No improvement in Val Score for {patience_counter} epochs.")

    # Early Stopping
    if patience_counter >= early_stopping_patience:
        logging.info("Early stopping triggered.")
        break

# ===============================
# 14. Load Best Model and Final Evaluation
# ===============================

model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

val_loss = 0
val_preds = []
val_targets = []

with torch.no_grad():
    progress_bar = tqdm(val_loader, desc='Final Evaluation', leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        val_preds.extend(preds)
        val_targets.extend(labels.detach().cpu().numpy())

avg_val_loss = val_loss / len(val_dataset)
try:
    val_auc = roc_auc_score(val_targets, val_preds)
except ValueError:
    val_auc = 0.0  # Handle cases where AUC cannot be computed

# Calculate accuracy
preds_binary = (np.array(val_preds) > 0.5).astype(int)
val_acc = accuracy_score(val_targets, preds_binary) * 100  # Convert to percentage

# Calculate custom score
custom_score = calculate_custom_score(val_targets, preds_binary)

logging.info("===== Final Evaluation =====")
logging.info(f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.2f}% | Val Score: {custom_score:.4f}")

# ===============================
# 15. Save the Final Model
# ===============================

torch.save(model.state_dict(), 'checkpoints/final_model.pth')
logging.info("Final model saved as 'checkpoints/final_model.pth'.")
