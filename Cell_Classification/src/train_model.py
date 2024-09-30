import json
import logging
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm


from data.make_dataset import get_dataloaders
from src.utils import calculate_custom_score
from configurations_training import set_seed, setup_logging, get_device, create_directories
from models.model import get_models, FocalLoss, get_model_parallel, objective

# =============================================================================================
# 1. Setup and Configuration
# =============================================================================================
set_seed(42)

# Setup logging
def setup_logging(log_dir: Path, log_file: str = "training.log") -> None:
    """
    Configure logging to file and console.

    Args:
        log_dir (Path): Directory where the log file will be saved.
        log_file (str, optional): Log file name. Defaults to "training.log".
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")
log_directory = Path("logs")
setup_logging(log_directory)

# Determine computation device
device, num_gpus = get_device()

# Create necessary directories
directories_to_create = ["plots", "checkpoints", "logs"]
create_directories(directories_to_create)

# Additional setup or training code can be added here
logging.info("Setup complete. Ready to proceed with training.")

# =============================================================================================
# 2. Get DataLoaders
# =============================================================================================
batch_size = 16  
img_size = 224  

# Get DataLoaders
train_loader, val_loader = get_dataloaders(batch_size, img_size)

# =============================================================================================
# 3. Get Objective Function
# =============================================================================================

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation custom score to maximize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters to tune
    model_name = trial.suggest_categorical('model_name', ['ViT', 'ResNet50', 'EfficientNet', 'MobileNetV3'])

    if model_name == 'ViT':
        img_size = 224  # ViT only supports 224x224 images
    else:
        img_size = trial.suggest_categorical('img_size', [224, 256, 299])

    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)

    num_epochs = 50 # Fixed number of epochs
    
    train_loader, val_loader = get_dataloaders(batch_size, img_size)

    # Get model
    models_dict = get_models()
    model = models_dict[model_name]
    model = get_model_parallel(model)
    model = model.to(device)

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

# =============================================================================================
# 4. Run Optuna Study
# =============================================================================================
n_trials = 2
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials, timeout=None)  

logging.info("========== Best Trial ==========")
best_trial = study.best_trial

logging.info(f"Value (Custom Score): {best_trial.value}")
logging.info(" Params: ")
for key, value in best_trial.params.items():
    logging.info(f"    {key}: {value}")

# =============================================================================================
# 5. Train Best Model with Best Hyperparameters
# =============================================================================================

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

    num_epochs = 10  # Increased epochs for final training

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
    # Print Final Summary
    # ===============================

    logging.info("===== Best Model Configuration =====")
    for key, value in model_info.items():
        logging.info(f"{key}: {value}")

    logging.info("===== Final Evaluation Metrics =====")
    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Validation Accuracy: {val_acc:.2f}%")
    logging.info(f"Custom Score: {custom_score:.4f}")

# ===============================
# 6. Execute Training of Best Model
# ===============================

# Train the best model
train_best_model(best_trial)
