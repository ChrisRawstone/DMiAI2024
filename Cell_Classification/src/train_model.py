import json
import logging
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from data.make_dataset import (
    get_dataloaders_final_train,
    LoadTifDataset,
    get_combined_dataloader
)
from src.utils import calculate_custom_score
from models.model import FocalLoss, BalancedBCELoss, get_model_parallel
from utils import get_model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_info(model_info_path):
    """
    Load model configuration from a JSON file and initialize the model.

    Args:
        model_info_path (str): Path to the JSON file containing model configuration.

    Returns:
        model (nn.Module): Initialized model.
        config (dict): Dictionary containing model configuration.
    """
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    model = get_model(model_name, num_classes=1)
    
    return model, model_info

def train_model(model, config, device, save_path_final, patience=10, final_train=False): 
    """
    Train the model with the provided configuration.

    Args:
        model (nn.Module): The model to train.
        config (dict): Dictionary containing training configuration.
        device (torch.device): The device to use for training.
        save_path_final (str): Path to save the final model.
        patience (int, optional): Early stopping patience. Defaults to 10.
        final_train (bool, optional): Whether to use combined dataset. Defaults to False.

    Returns:
        nn.Module: Trained model.
    """
    img_size = config['img_size']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    gamma = config['gamma']
    alpha = config['alpha']
    loss_function = config['loss_function']
    optimizer_name = config['Optimizer']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    alpha_optim = config.get('alpha_optim', 0.99)  # Default value if not provided

    # Get dataloaders
    if final_train:
        train_loader = get_combined_dataloader(batch_size, img_size)
    else:
        train_loader, _, _ = get_dataloaders_final_train(batch_size, img_size)
        train_dataset = LoadTifDataset(
            image_dir='data/training',
            csv_file_path='data/training.csv',
            transform=None
        )
        # Extract labels from the full dataset
        labels = train_dataset.labels_df.iloc[:, 1].values.astype(int)  # Assuming labels are in the second column
        # Compute class counts and pos_weight for Weighted BCE Loss
        class_counts = Counter(labels)
        num_positive = class_counts.get(1, 0)
        num_negative = class_counts.get(0, 0)
        pos_weight = num_negative / num_positive if num_positive > 0 else 1.0

    # Define loss function
    if loss_function == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif loss_function == 'WeightedBCEWithLogitsLoss':
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
    elif loss_function == 'FocalLoss':
        criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True, reduce=True).to(device)
    elif loss_function == 'BalancedCrossEntropyLoss':
        criterion = BalancedBCELoss().to(device)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    # Load model to device and set up for parallel training if multiple GPUs are available
    model = model.to(device)
    model = get_model_parallel(model)
    
    # Create optimizer
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=alpha_optim,
            eps=epsilon,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Learning rate scheduler and gradient scaler for mixed precision
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast('cuda'):  
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale the loss and perform backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            # Collect predictions and targets for metrics
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

            progress_bar.set_postfix({'Loss': loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation Phase on Training Data
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            progress_bar = tqdm(train_loader, desc='Validation on Training Data', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.detach().cpu().numpy())

                progress_bar.set_postfix({'Val Loss': loss.item()})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(train_loader.dataset)

        # Calculate metrics
        preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_acc = accuracy_score(val_targets, preds_binary) * 100 
        custom_score = calculate_custom_score(val_targets, preds_binary)

        # Log epoch metrics
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Custom Score: {custom_score:.4f} | "
            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                break

        # Update the learning rate scheduler
        scheduler.step()
    
    # Save the final model
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path_final)
    else:
        torch.save(model.state_dict(), save_path_final)
    
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}.")

    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the provided data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): The device to use for evaluation.
    """
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    # Calculate metrics
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    custom_score = calculate_custom_score(val_targets, preds_binary)

    ground_truths = np.array(val_targets)
    predictions = np.array(preds_binary)

    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, zero_division=0)
    recall = recall_score(ground_truths, predictions, zero_division=0, average='macro')
    f1 = f1_score(ground_truths, predictions, zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Custom Score : {custom_score:.4f}")

if __name__ == "__main__":
    model_info_path = 'MODELS_FINAL_DEPLOY/best_model_5.json'
    save_path_final = 'MODELS_FINAL_EVAL/best_trained_model_6.pth'
    save_path_final_all = 'MODELS_FINAL_DEPLOY/best_trained_model_5.pth'

    # Load model and configuration
    model_loaded, config = load_model_and_info(model_info_path)

    # Train the model
    model_trained = train_model(
        model_loaded,
        config,
        device,
        save_path_final
    )

    # Load the model again for final training
    model_loaded_final, config = load_model_and_info(model_info_path)
    config['num_epochs'] += 5  # Increase the number of epochs for final training

    # Final training on combined dataset
    model_final = train_model(
        model_loaded_final,
        config,
        device,
        save_path_final_all,
        final_train=True
    )

    # Get data loaders for evaluation
    batch_size = config['batch_size']
    img_size = config['img_size']
    train_loader, test_loader, train_loader_eval = get_dataloaders_final_train(batch_size, img_size)
    train_loader_all = get_combined_dataloader(batch_size, img_size)

    # Evaluate the model
    print('Evaluation on test data:')
    evaluate_model(model_trained, test_loader, device)

    print('Evaluation on training data:')
    evaluate_model(model_trained, train_loader_eval, device)

    print('Final model evaluation on combined data:')
    evaluate_model(model_final, train_loader_all, device)