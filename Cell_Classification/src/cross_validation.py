# train_model.py

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import logging
import gc
from models.model import get_models, FocalLoss, get_model_parallel
from data.make_dataset import get_dataloaders
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import logging
from sklearn.model_selection import StratifiedKFold
from threading import Lock
import optuna
import os
import torch
import logging
from datetime import datetime
from threading import Lock
from src.utils import calculate_custom_score
from utils import get_model, calculate_custom_score
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from typing import Tuple
import logging
import numpy as np
import torch
from tqdm import tqdm

from src.data.make_dataset import LoadTifDataset, get_dataloaders, get_transforms, setup_working_directory

# def load_model_and_info(model_info_path):
#     with open(model_info_path, 'r') as f:
#         model_info = json.load(f)
    
#     model_name = model_info['model_name']
#     img_size = model_info['img_size']
#     batch_size = model_info['batch_size']
#     num_epochs = model_info['num_epochs']
#     learning_rate = model_info['learning_rate']
#     weight_decay = model_info['weight_decay']
#     gamma = model_info['gamma']
#     alpha = model_info['alpha']

#     model = get_model(model_name, num_classes=1)    
#     return model, img_size, batch_size, num_epochs, learning_rate, weight_decay, gamma, alpha


# def get_cross_val_loaders(
#     batch_size: int,
#     img_size: int,
#     n_splits: int = 5,
#     fold: int = 0,
#     shuffle: bool = True,
#     random_state: int = 42):

#     combined_image_dir = Path("data/combined")
#     combined_csv_path = Path("data/combined.csv")
    
#     # Get transforms
#     train_transform, val_transform = get_transforms(img_size)
    
#     # Create the dataset with train_transform initially; we'll handle transforms separately later
#     dataset = LoadTifDataset(
#         image_dir=combined_image_dir,
#         csv_file_path=combined_csv_path,
#         transform=None)
    
#     # Read combined CSV to get labels for stratification if needed
#     combined_df = pd.read_csv(combined_csv_path)
#     labels = combined_df['is_homogenous'].values  # Adjust the column name if necessary
    
#     # Initialize KFold or StratifiedKFold
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
#     # Get all indices
#     indices = np.arange(len(dataset))
    
#     # Split the data into folds
#     for current_fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
#         if current_fold == fold:
#             train_subset = Subset(
#                 LoadTifDataset(
#                     image_dir=combined_image_dir,
#                     csv_file_path=combined_csv_path,
#                     transform=train_transform
#                 ),
#                 train_idx
#             )
#             val_subset = Subset(
#                 LoadTifDataset(
#                     image_dir=combined_image_dir,
#                     csv_file_path=combined_csv_path,
#                     transform=val_transform
#                 ),
#                 val_idx
#             )
            
#             # Create DataLoaders
#             train_loader = DataLoader(
#                 train_subset,
#                 batch_size=batch_size,
#                 shuffle=True,  # Shuffle training data
#                 num_workers=8,
#                 pin_memory=True
#             )
            
#             val_loader = DataLoader(
#                 val_subset,
#                 batch_size=batch_size,
#                 shuffle=False,  # Do not shuffle validation data
#                 num_workers=8,
#                 pin_memory=True
#             )
            
#             return train_loader, val_loader


# def train_model(model, alpha, gamma, lr, weight_decay, num_epochs, train_loader, val_loader, device='cuda'):
    
#     # Loss function
#     criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

#     # Optimizer and Scheduler
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

#     # Mixed Precision Scaler
#     scaler = torch.amp.GradScaler()

#     for epoch in range(num_epochs):
#         logging.info(f"Epoch {epoch+1}/{num_epochs}")

#         # Training Phase
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_targets = []

#         for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False):
#             images = images.to(device, non_blocking=True)
#             labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#             optimizer.zero_grad()

#             with torch.amp.autocast("cuda"):
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             train_loss += loss.item() * images.size(0)
#             preds = torch.sigmoid(outputs).detach().cpu().numpy()
#             train_preds.extend(preds)
#             train_targets.extend(labels.detach().cpu().numpy())

#         # Validation Phase
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_targets = []

#         with torch.no_grad():
#             for images, labels in tqdm(val_loader, desc='Validation', leave=False):
#                 images = images.to(device, non_blocking=True)
#                 labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

#                 with torch.amp.autocast("cuda"):
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)

#                 val_loss += loss.item() * images.size(0)
#                 preds = torch.sigmoid(outputs).detach().cpu().numpy()
#                 val_preds.extend(preds)
#                 val_targets.extend(labels.detach().cpu().numpy())

#         preds_binary = (np.array(val_preds) > 0.5).astype(int)
#         custom_score = calculate_custom_score(val_targets, preds_binary)

#         scheduler.step()
        
#         return custom_score

    
# def train_cv(model, device='cuda'):
   
#     # Load model and info
#     model, img_size, batch_size, num_epochs, learning_rate, weight_decay, gamma, alpha = load_model_and_info(model, device)
    
#     # Cross-Validation Training Loop
#     n_splits = 5
#     batch_size = 32
#     img_size = 224
#     num_epochs = 10
#     random_state = 42
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for fold in range(n_splits):
        
#         # Get DataLoaders for the current fold
#         train_loader, val_loader = get_cross_val_loaders(
#             batch_size=batch_size,
#             img_size=img_size,
#             n_splits=n_splits,
#             fold=fold,
#             shuffle=True,
#             random_state=random_state)
    
        
#         print("DataLoaders created")

# train_cv('checkpoints/best_model_optuna.json')



#     # Loss function
#     criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

#     # Optimizer and Scheduler
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

#     # Mixed Precision Scaler
#     scaler = torch.amp.GradScaler("cuda")  # Updated to use torch.cuda.amp

#     best_custom_score = 0
#     early_stopping_patience = 10
#     patience_counter = 0

#     # Initialize model_info dictionary
#     model_info = {
#         'model_name': model_name,
#         'img_size': img_size,
#         'batch_size': batch_size,
#         'epochs': num_epochs,
#         'learning_rate': lr,
#         'weight_decay': weight_decay,
#         'gamma': gamma,
#         'alpha': alpha
#     }

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

#             with torch.amp.autocast("cuda"):  # Updated autocast usage
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

#                 with torch.amp.autocast("cuda"):  # Updated autocast usage
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
#             # Save the best model from final training with a different name
#             save_path_final = 'checkpoints/best_model_final.pth'  # Changed filename
#             if isinstance(model, nn.DataParallel):
#                 torch.save(model.module.state_dict(), save_path_final)
#             else:
#                 torch.save(model.state_dict(), save_path_final)
#             logging.info(f"Best final model saved with Val Score: {best_custom_score:.4f}")
#             patience_counter = 0

#             # Save model architecture name
#             model_info_current = {
#                 'model_name': model_name,
#                 'img_size': img_size
#             }
#             with open('checkpoints/model_info_final.json', 'w') as f:
#                 json.dump(model_info_current, f)
#             logging.info("Final model architecture information saved to 'checkpoints/model_info_final.json'.")

#         else:
#             patience_counter += 1
#             logging.info(f"No improvement in Val Score for {patience_counter} epochs.")

#         # Early Stopping
#         if patience_counter >= early_stopping_patience:
#             logging.info("Early stopping triggered.")
#             break

#     # Load best model
#     if isinstance(model, nn.DataParallel):
#         model.module.load_state_dict(torch.load('checkpoints/best_model_final.pth'))
#     else:
#         model.load_state_dict(torch.load('checkpoints/best_model_final.pth'))
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

#             with torch.amp.autocast("cuda"):  # Updated autocast usage
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
#     save_path_final = 'checkpoints/final_model_final.pth'  # Changed filename to differentiate from Optuna best
#     if isinstance(model, nn.DataParallel):
#         torch.save(model.module.state_dict(), save_path_final)
#     else:
#         torch.save(model.state_dict(), save_path_final)
#     logging.info(f"Final model saved as '{save_path_final}'.")

#     # Optionally, save the model architecture info for the final model
#     with open('checkpoints/final_model_info_final.json', 'w') as f:
#         json.dump(model_info, f)
#     logging.info("Final model architecture information saved to 'checkpoints/final_model_info_final.json'.")

#     # ===============================
#     # Print Final Summary
#     # ===============================

#     logging.info("===== Best Model Configuration =====")
#     for key, value in model_info.items():
#         logging.info(f"{key}: {value}")

#     logging.info("===== Final Evaluation Metrics =====")
#     logging.info(f"Validation Loss: {avg_val_loss:.4f}")
#     logging.info(f"Validation Accuracy: {val_acc:.2f}%")
#     logging.info(f"Custom Score: {custom_score:.4f}")

# # ===============================
# # 6. Execute Training of Best Model
# # ===============================

# # Train the best model
# train_best_model(best_trial)

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def load_model_and_info(model_info_path):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']
    batch_size = model_info['batch_size']
    #num_epochs = model_info['num_epochs']
    learning_rate = model_info['learning_rate']
    weight_decay = model_info['weight_decay']
    gamma = model_info['gamma']
    alpha = model_info['alpha']

    model = get_model(model_name, num_classes=1)    
    return model, img_size, batch_size, learning_rate, weight_decay, gamma, alpha, model_name #, num_epochs

def get_cross_val_loaders(
    batch_size: int,
    img_size: int,
    n_splits: int = 3,
    fold: int = 0,
    shuffle: bool = True,
    random_state: int = 42):

    combined_image_dir = Path("data/combined")
    combined_csv_path = Path("data/combined.csv")
    
    # Get transforms
    train_transform, val_transform = get_transforms(img_size)
    
    # Create the dataset
    dataset = LoadTifDataset(
        image_dir=combined_image_dir,
        csv_file_path=combined_csv_path,
        transform=None)
    
    # Read labels for stratification
    combined_df = pd.read_csv(combined_csv_path)
    labels = combined_df['is_homogenous'].values  # Adjust the column name if necessary
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Get all indices
    indices = np.arange(len(dataset))
    
    # Split the data into folds
    for current_fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        if current_fold == fold:
            train_subset = Subset(
                LoadTifDataset(
                    image_dir=combined_image_dir,
                    csv_file_path=combined_csv_path,
                    transform=train_transform
                ),
                train_idx
            )
            val_subset = Subset(
                LoadTifDataset(
                    image_dir=combined_image_dir,
                    csv_file_path=combined_csv_path,
                    transform=val_transform
                ),
                val_idx
            )
            
            # Create DataLoaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,  # Shuffle training data
                num_workers=8,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,  # Do not shuffle validation data
                num_workers=8,
                pin_memory=True
            )
            
            return train_loader, val_loader

def train_model(model, alpha, gamma, lr, weight_decay, num_epochs, train_loader, val_loader, device='cuda', fold=0):
    # Early Stopping Parameters
    num_epochs = 100  # Number of epochs to train
    patience = 20  # Number of epochs to wait for improvement before stopping
    best_score = None
    epochs_no_improve = 0

    # Create a directory to save models
    os.makedirs(f'models/fold_{fold}', exist_ok=True)

    # Logger setup
    logger = logging.getLogger(f'Fold_{fold}')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'models/fold_{fold}/training.log', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)

    # Loss function
    criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Training Phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

        # Calculate average train loss
        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.detach().cpu().numpy())

        # Calculate average val loss
        val_loss /= len(val_loader.dataset)

        # Calculate custom score
        preds_binary = (np.array(val_preds) > 0.5).astype(int)
        custom_score = calculate_custom_score(val_targets, preds_binary)

        logger.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Custom Score {custom_score:.4f}')
        
        # Check for improvement
        if best_score is None or custom_score > best_score:
            best_score = custom_score
            epochs_no_improve = 0
            # Save the model
            torch.save(model.state_dict(), f'models/fold_{fold}/best_model.pth')
            logger.info(f'Best model saved at epoch {epoch+1}')
            logging.info(f"Fold {fold}, epoch {epoch+1}, Custom Score: {best_score}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

        scheduler.step()

    return custom_score

def train_cv(model_info_path, device='cuda'):
    # Load model and info
    _, img_size, batch_size, learning_rate, weight_decay, gamma, alpha, model_name = load_model_and_info(model_info_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Cross-Validation Training Loop
    n_splits = 3
    random_state = 42
    num_epochs = 100
    for fold in range(n_splits):
        # Get DataLoaders for the current fold
        train_loader, val_loader = get_cross_val_loaders(
            batch_size=batch_size,
            img_size=img_size,
            n_splits=n_splits,
            fold=fold,
            shuffle=True,
            random_state=random_state)

        # Create a fresh instance of the model for each fold
        model_fold = get_model(model_name, num_classes=1)
        model_fold = model_fold.to(device)

        # Train the model for this fold
        custom_score = train_model(
            model_fold,
            alpha, gamma,
            learning_rate,
            weight_decay,
            num_epochs,
            train_loader,
            val_loader,
            device=device,
            fold=fold)

        # Save the model after each fold
        torch.save(model_fold.state_dict(), f"models/fold_{fold}/final_model.pth")

        # Log the custom score
        logging.info(f"Fold {fold}, Custom Score: {custom_score}")

# Start cross-validation training
setup_working_directory()
train_cv('checkpoints/model_info_optuna.json')
