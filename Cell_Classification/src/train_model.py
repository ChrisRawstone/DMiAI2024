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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
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
# 2. Data Decoding Functions
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
# 3. Custom Dataset
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
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = transforms.ToTensor()(image).permute(2, 0, 1)  # Convert to [C, H, W]

        return image, label

# ===============================
# 4. Data Augmentation and Transforms
# ===============================

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Updated for 3 channels
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Updated for 3 channels
    ToTensorV2(),
])

# ===============================
# 5. Load CSV Files
# ===============================

train_df = pd.read_csv('data/training.csv')
val_df = pd.read_csv('data/validation.csv')

logging.info(f"Training samples: {len(train_df)}")
logging.info(f"Validation samples: {len(val_df)}")

# ===============================
# 6. Create Datasets and DataLoaders
# ===============================

train_dataset = CustomDataset(train_df, 'data/training8bit', transform=train_transform)
val_dataset = CustomDataset(val_df, 'data/validation', transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
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
# 7. Save Sample Images
# ===============================

def save_sample_images(dataset, num_samples=5, folder='plots', split='train'):
    os.makedirs(folder, exist_ok=True)
    for i in range(num_samples):
        image, label = dataset[i]
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 0.5) + 0.5  # Unnormalize
        image_np = np.clip(image_np, 0, 1)
        plt.imshow(image_np)
        plt.title(f'{split.capitalize()} Label: {label}')
        plt.axis('off')
        plt.savefig(f'{folder}/{split}_sample_{i}.png')
        plt.close()

save_sample_images(train_dataset, split='train')
save_sample_images(val_dataset, split='val')

logging.info("Sample images saved to 'plots' folder.")

# ===============================
# 8. Define the Model
# ===============================

def get_model(pretrained=True):
    model = models.efficientnet_b4(pretrained=pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)  # Binary classification
    return model

model = get_model(pretrained=True)
model = model.to(device)

# ===============================
# 9. Define Focal Loss with Class Weights
# ===============================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
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

# Calculate class weights if there's imbalance
class_counts = train_df['is_homogenous'].value_counts().to_dict()
total = sum(class_counts.values())
class_weights = {
    0: total / class_counts[0],
    1: total / class_counts[1]
}
alpha = class_weights[1] / (class_weights[0] + class_weights[1])

criterion = FocalLoss(alpha=alpha, gamma=2, logits=True).to(device)

# ===============================
# 10. Optimizer and Scheduler
# ===============================

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ===============================
# 11. Mixed Precision Scaler
# ===============================

scaler = torch.cuda.amp.GradScaler()

# ===============================
# 12. Training and Validation Loops
# ===============================

num_epochs = 50
best_val_acc = 0
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
    train_auc = roc_auc_score(train_targets, train_preds)

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    correct = 0
    total = 0

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
    val_auc = roc_auc_score(val_targets, val_preds)

    # Calculate accuracy
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    val_acc = accuracy_score(val_targets, preds_binary)

    logging.info(f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")
    logging.info(f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.2f}%")

    # Scheduler step
    scheduler.step()

    # Checkpointing
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        logging.info(f"Best model saved with Val Acc: {best_val_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        logging.info(f"No improvement in Val Acc for {patience_counter} epochs.")

    # Early Stopping
    if patience_counter >= early_stopping_patience:
        logging.info("Early stopping triggered.")
        break

# ===============================
# 13. Load Best Model and Final Evaluation
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
val_auc = roc_auc_score(val_targets, val_preds)
preds_binary = (np.array(val_preds) > 0.5).astype(int)
val_acc = accuracy_score(val_targets, preds_binary)

logging.info("===== Final Evaluation =====")
logging.info(f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.2f}%")

# ===============================
# 14. Save the Final Model
# ===============================

torch.save(model.state_dict(), 'checkpoints/final_model.pth')
logging.info("Final model saved as 'checkpoints/final_model.pth'.")

# ===============================
# 15. Optional: TensorBoard Logging
# ===============================
# Uncomment the following lines if you wish to use TensorBoard for logging.

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='logs/tensorboard')

# # Inside the training loop, add:
# writer.add_scalar('Loss/Train', avg_train_loss, epoch)
# writer.add_scalar('Loss/Val', avg_val_loss, epoch)
# writer.add_scalar('AUC/Train', train_auc, epoch)
# writer.add_scalar('AUC/Val', val_auc, epoch)
# writer.add_scalar('Accuracy/Val', val_acc, epoch)

# # After training
# writer.close()
