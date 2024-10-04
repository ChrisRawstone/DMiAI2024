# train_model.py
import copy
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import Counter
from data.make_dataset import get_dataloaders_final_train, LoadTifDataset, get_dataloaders, get_dataloaders_train, get_combined_dataloader
from src.utils import calculate_custom_score
from models.model import FocalLoss, BalancedBCELoss, get_model_parallel
from utils import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

def load_model_and_info(model_info_path):
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    img_size = model_info['img_size']
    batch_size = model_info['batch_size']
    num_epochs = model_info['num_epochs']
    lr = model_info['learning_rate']
    weight_decay = model_info['weight_decay']
    gamma = model_info['gamma']
    alpha = model_info['alpha']
    loss_function = model_info['loss_function']
    optimizer_name = model_info['Optimizer']
    beta1 = model_info['beta1']
    beta2 = model_info['beta2']
    epsilon = model_info['epsilon']
    alpha_optim = model_info['alpha_optim']
        
    model = get_model(model_name, num_classes=1)    
    
    return model, model_name, img_size, batch_size, lr, weight_decay, gamma, alpha, num_epochs, loss_function, optimizer_name, beta1, beta2, epsilon, alpha_optim


def train_model(model, img_size, batch_size, lr, weight_decay, gamma, alpha, num_epochs, loss_function, optimizer_name, 
               beta1, beta2, epsilon, alpha_optim, device, save_path_final, patience=10, final_train=False): 

    # Get dataloaders
    # train_loader, val_loader = get_dataloaders_train(batch_size, img_size)
    if final_train:
        train_loader = get_combined_dataloader(batch_size, img_size)
    
    else:
        train_loader, _, _ = get_dataloaders_final_train(batch_size, img_size)
        train_dataset = LoadTifDataset(
        image_dir='data/training',
        csv_file_path='data/training.csv',
        transform=None )

        # Extract labels from the full dataset
        labels = train_dataset.labels_df.iloc[:, 1].values.astype(int)  # Assuming labels are in the second column

        # Compute class counts and pos_weight for Weighted BCE Loss
        class_counts = Counter(labels)
        num_positive = class_counts[1] if 1 in class_counts else 0
        num_negative = class_counts[0] if 0 in class_counts else 0
        if num_positive == 0 or num_negative == 0:
            pos_weight = 1.0  # Avoid division by zero; adjust as needed
        else:
            pos_weight = num_negative / num_positive

    # Loss function    
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

    # Load model 
    model = model.to(device)
    model = get_model_parallel(model)
    
    # Create optimizer using the sampled hyperparameters
    if optimizer_name in ['AdamW', 'Adam']:
        optimizer_class = optim.AdamW if optimizer_name == 'AdamW' else optim.Adam
        optimizer = optimizer_class(
            model.parameters(),
            lr=lr,
            betas=(beta1,beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            #alpha=alpha_optim,
            eps=epsilon,
            weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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

        # Calculate accuracy
        preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_acc = accuracy_score(val_targets, preds_binary) * 100 

        # Calculate custom score
        custom_score = calculate_custom_score(val_targets, preds_binary)

        # Log epoch metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Custom Score: {custom_score:.4f} | "
            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
        else:
            epochs_no_improve += 1

        # # Check if early stopping should be triggered
        # if epochs_no_improve >= patience:
        #     logging.info(f"Early stopping triggered after {epoch+1} epochs.")
        #     break

        # Update the learning rate scheduler
        scheduler.step()
    
    # Save the final model
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path_final)
    else:
        torch.save(model.state_dict(), save_path_final)
    

    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}.")

    return model


def evaluate_model(model, test_loader, device):
    
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='test', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):  # Updated autocast usage
                outputs = model(images)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    # Calculate custom score
    preds_binary = (np.array(val_preds) > 0.5).astype(int)
    custom_score = calculate_custom_score(val_targets, preds_binary)

    ground_truths = np.array(val_targets)
    predictions = np.array(preds_binary)

    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, zero_division=0)
    recall = recall_score(ground_truths, predictions, zero_division=0, average='macro')
    f1 = f1_score(ground_truths, predictions, zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Custom Score: {custom_score:.4f}")


if __name__ == "__main__":
    model_info = 'MODELS_FINAL_DEPLOY/best_model_5.json'
    
    save_path_final = 'MODELS_FINAL_EVAL/best_trained_model_6.pth'
    save_path_final_all = 'MODELS_FINAL_DEPLOY/best_trained_model_5.pth'
    
    model_loaded, model_name, img_size, batch_size, lr, weight_decay, gamma, alpha, num_epochs, loss_function, optimizer_name, beta1, beta2, epsilon, alpha_optim = load_model_and_info(model_info)
    
    model_trained = train_model(model_loaded, img_size, batch_size, lr, weight_decay, gamma, alpha, 
                                num_epochs, loss_function, optimizer_name, beta1, beta2, epsilon, alpha_optim, device, save_path_final)
    
    # model = get_model(model_name, num_classes=1)
    # checkpoint = torch.load('MODELS_FINAL_EVAL/best_trained_model_8.pth', map_location=device, weights_only=True)
    # model.load_state_dict(checkpoint)
    # model.to(device)
    
    model_loaded_final = get_model(model_name, num_classes=1)
    model_loaded_final, model_name, img_size, batch_size, lr, weight_decay, gamma, alpha, num_epochs, loss_function, optimizer_name, beta1, beta2, epsilon, alpha_optim = load_model_and_info(model_info)
    # checkpoint = torch.load('MODELS_FINAL_DEPLOY/best_trained_model_8.pth', map_location=device, weights_only=True)
    # model_loaded_final.load_state_dict(checkpoint)
    # model_loaded_final.to(device)
    num_epochs += 5
    model_final = train_model(model_loaded_final, img_size, batch_size, lr, weight_decay, gamma, alpha, num_epochs, loss_function, optimizer_name, beta1, beta2, 
                                          epsilon, alpha_optim, device, save_path_final_all, final_train=True)
        
    train_loader, test_loader, train_loader_eval = get_dataloaders_final_train(batch_size, img_size)
    train_loader_all = get_combined_dataloader(batch_size, img_size)
    
    print('on test')
    evaluate_model(model_trained, test_loader, device)
    
    print('on training')
    evaluate_model(model_trained, train_loader_eval, device)
    
    print('final model on all')
    evaluate_model(model_final, train_loader_all, device)
    

    # model = get_model(model_name, num_classes=1)
    # checkpoint = torch.load('checkpoints/best_model_optuna.pth', map_location=device, weights_only=True)
    # model.load_state_dict(checkpoint)
    # model.to(device)

